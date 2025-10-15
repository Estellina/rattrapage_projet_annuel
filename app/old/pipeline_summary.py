from __future__ import annotations
import logging, time, pathlib
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app import utils
from app.config import cfg

# Initialisation logger
log = logging.getLogger("summary_pipeline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)

# Téléchargement S3 (tolérant)
try:
    from app.s3_loader import download_s3_prefix
except Exception:
    try:
        from app.s3_utils import download_s3_prefix
    except Exception:
        def download_s3_prefix(prefix: str, dst: pathlib.Path):
            return  # no-op si non dispo

# Cache pour le modèle pré-entraîné mBART
_PT_READY = False
_PT_MODEL = None
_PT_TOKENIZER = None
_PT_SOURCE = "unknown"

def _fix_generation_config(model):
    gen = getattr(model, "generation_config", None)
    if gen and getattr(gen, "early_stopping", None) is None:
        gen.early_stopping = True
    try:
        model.config.use_cache = True
    except Exception:
        pass

def _deviceize(model: AutoModelForSeq2SeqLM):
    if torch.cuda.is_available():
        try:
            model.half()
        except Exception:
            pass
        model.to("cuda").eval()
    else:
        model.to("cpu").eval()

def _ensure_pt():
    """Charge un mBART depuis le cache (ou S3), sinon fallback HF."""
    global _PT_READY, _PT_MODEL, _PT_TOKENIZER, _PT_SOURCE
    if _PT_READY:
        return

    cache = pathlib.Path(cfg.CACHE_DIR) / "summary_pt_mbart"
    cache.mkdir(parents=True, exist_ok=True)
    # Récupération depuis S3 seulement si les poids n'existent pas
    has_weights = (cache / "model.safetensors").exists() or (cache / "pytorch_model.bin").exists()
    if getattr(cfg, "PT_SUMMARY_S3_PREFIX", None) and not has_weights:
        try:
            download_s3_prefix(cfg.PT_SUMMARY_S3_PREFIX, cache)
        except Exception as e:
            log.warning(f"[SUMMARY][PT] S3 download skipped: {e}")

    # Tentative locale
    try:
        tok = AutoTokenizer.from_pretrained(str(cache), use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(str(cache))
        # Langue cible mBART
        if hasattr(tok, "lang_code_to_id"):
            tok.src_lang = "en_XX"
            forced = tok.lang_code_to_id.get("fr_XX")
            if forced is not None:
                mdl.config.forced_bos_token_id = forced
        _fix_generation_config(mdl)
        _deviceize(mdl)
        _PT_TOKENIZER, _PT_MODEL, _PT_SOURCE = tok, mdl, "local"
        _PT_READY = True
        log.info(f"[SUMMARY][PT] chargé localement")
        return
    except Exception as e:
        log.warning(f"[SUMMARY][PT] chargement local invalide: {e!s}")

    # Fallback HuggingFace
    model_id = getattr(cfg, "PT_SUMMARY_HF_MODEL", "facebook/mbart-large-50")
    log.warning(f"[SUMMARY][PT] fallback HF: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if hasattr(tok, "lang_code_to_id"):
        tok.src_lang = "en_XX"
        forced = tok.lang_code_to_id.get("fr_XX")
        if forced is not None:
            mdl.config.forced_bos_token_id = forced
    _fix_generation_config(mdl)
    _deviceize(mdl)
    _PT_TOKENIZER, _PT_MODEL, _PT_SOURCE = tok, mdl, f"hf://{model_id}"
    _PT_READY = True

def run_summary_pipeline(pdf_bytes: bytes) -> Dict[str, any]:
    # Extraction du texte
    start = time.time()
    text = utils.extract_text_safe(pdf_bytes)
    if not text or not text.strip():
        raise ValueError("Le PDF ne contient pas de texte exploitable.")
    log.info(f"[SUMMARY] extraction ok ({len(text)} chars) in {time.time()-start:.2f}s")

    # Nettoyage + troncature dure
    max_chars = getattr(cfg, "SUMMARY_MAX_CHARS", 20000)
    if len(text) > max_chars:
        text = text[:max_chars]
    cleaned = " ".join(text.split())

    # Traduction FR->EN (offline Marian)
    start = time.time()
    english, _ = utils.maybe_translate(
        cleaned,
        target_lang=getattr(cfg, "TRANSLATION_TGT", "en"),
        enable_offline=getattr(cfg, "ENABLE_OFFLINE_TRANSLATION", True),
    )
    log.info(f"[SUMMARY] traduction in EN in {time.time()-start:.2f}s")

    # Chargement du modèle PT (local ou HF)
    _ensure_pt()

    # Génération chunkée
    start = time.time()
    max_tokens = getattr(cfg, "SUMMARY_PT_MAX_TOKENS", 900)
    overlap    = getattr(cfg, "SUMMARY_PT_OVERLAP", 150)
    gen_kwargs = dict(max_new_tokens=224, num_beams=2, early_stopping=True)

    summary_fr = utils.chunked_generate_text(
        _PT_MODEL, _PT_TOKENIZER, english,
        max_tokens=max_tokens, overlap=overlap,
        gen_kwargs=gen_kwargs
    )
    log.info(f"[SUMMARY] génération PT in {time.time()-start:.2f}s (src={_PT_SOURCE})")

    # Score de qualité
    score = utils.simple_quality_score(cleaned, summary_fr or "")
    return {
        "summary_fr": (summary_fr or "").strip(),
        "summary_en": "",  # en version actuelle on ne retourne que le français
        "model_used": f"pt_mbart:{_PT_SOURCE}",
        "quality": float(score),
    }
