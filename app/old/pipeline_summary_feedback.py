from __future__ import annotations
import logging, time, pathlib
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app import utils
from app.config import cfg

log = logging.getLogger("summary_feedback_pipeline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)

_PT_READY = False
_PT_MODEL = None
_PT_TOKENIZER = None
_PT_SOURCE = "unknown"

# Helpers (similaires au pipeline résumé)
def _fix_gen(model):
    gen = getattr(model, "generation_config", None)
    if gen and getattr(gen, "early_stopping", None) is None:
        gen.early_stopping = True
    try: model.config.use_cache = True
    except Exception: pass

def _deviceize(model):
    if torch.cuda.is_available():
        try: model.half()
        except Exception: pass
        model.to("cuda").eval()
    else:
        model.to("cpu").eval()

def _ensure_pt():
    global _PT_READY, _PT_MODEL, _PT_TOKENIZER, _PT_SOURCE
    if _PT_READY:
        return
    cache = pathlib.Path(cfg.CACHE_DIR) / "summary_pt_mbart"
    cache.mkdir(parents=True, exist_ok=True)
    has_weights = (cache / "model.safetensors").exists() or (cache / "pytorch_model.bin").exists()
    # Téléchargement s3 si besoin
    try:
        from app.s3_loader import download_s3_prefix
        if getattr(cfg, "PT_SUMMARY_S3_PREFIX", None) and not has_weights:
            download_s3_prefix(cfg.PT_SUMMARY_S3_PREFIX, cache)
    except Exception:
        pass

    # Tentative locale
    try:
        tok = AutoTokenizer.from_pretrained(str(cache), use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(str(cache))
        if hasattr(tok, "lang_code_to_id"):
            tok.src_lang = "en_XX"
            forced = tok.lang_code_to_id.get("fr_XX")
            if forced is not None:
                mdl.config.forced_bos_token_id = forced
        _fix_gen(mdl); _deviceize(mdl)
        _PT_TOKENIZER, _PT_MODEL, _PT_SOURCE = tok, mdl, "local"
        _PT_READY = True
        return
    except Exception:
        pass
    # fallback HF
    model_id = getattr(cfg, "PT_SUMMARY_HF_MODEL", "facebook/mbart-large-50")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if hasattr(tok, "lang_code_to_id"):
        tok.src_lang = "en_XX"
        forced = tok.lang_code_to_id.get("fr_XX")
        if forced is not None:
            mdl.config.forced_bos_token_id = forced
    _fix_gen(mdl); _deviceize(mdl)
    _PT_TOKENIZER, _PT_MODEL, _PT_SOURCE = tok, mdl, f"hf://{model_id}"
    _PT_READY = True

def _build_prefix(length="standard", ton="neutre", focus="général",
                  structure="paragraphes", couverture="concis",
                  style="abstractive", chiffres="garder", citations="inclure") -> str:
    # même mapping que dans le notebook (peut être extrait dans utils)
    LENGTH2TOKEN = {"court":"<LEN_SHORT>","standard":"<LEN_MEDIUM>","long":"<LEN_LONG>"}
    TONE2TOKEN   = {"formel":"<TONE_FORMAL>","neutre":"<TONE_NEUTRAL>","informel":"<TONE_CASUAL>"}
    FOCUS2TOKEN  = {
        "général":"<FOCUS_GENERAL>","détails":"<FOCUS_DETAILS>","résultats":"<FOCUS_RESULTS>",
        "méthodes":"<FOCUS_METHODS>","limites":"<FOCUS_LIMITATIONS>","applications":"<FOCUS_APPLICATIONS>"
    }
    STRUCT2TOKEN = {"paragraphes":"<STRUCT_PARAGRAPHS>","puces":"<STRUCT_BULLETS>","sections":"<STRUCT_SECTIONS>"}
    COVER2TOKEN  = {"concis":"<COVER_KEYPOINTS>","complet":"<COVER_COMPREHENSIVE>"}
    STYLE2TOKEN  = {"abstractive":"<STYLE_ABSTRACTIVE>","extractive":"<STYLE_EXTRACTIVE>"}
    NUM2TOKEN    = {"garder":"<NUM_KEEP>","réduire":"<NUM_MINIMIZE>"}
    CITE2TOKEN   = {"inclure":"<CITE_INCLUDE>","exclure":"<CITE_EXCLUDE>"}
    toks = [
        LENGTH2TOKEN.get(length,"<LEN_MEDIUM>"),
        TONE2TOKEN.get(ton,"<TONE_NEUTRAL>"),
        FOCUS2TOKEN.get(focus,"<FOCUS_GENERAL>"),
        STRUCT2TOKEN.get(structure,"<STRUCT_PARAGRAPHS>"),
        COVER2TOKEN.get(couverture,"<COVER_KEYPOINTS>"),
        STYLE2TOKEN.get(style,"<STYLE_ABSTRACTIVE>"),
        NUM2TOKEN.get(chiffres,"<NUM_KEEP>"),
        CITE2TOKEN.get(citations,"<CITE_INCLUDE>"),
        "<DOC_START>",
    ]
    return " ".join(toks) + " "

def run_summary_feedback_pipeline(pdf_bytes: bytes, prefs: Optional[Dict[str, str]] = None) -> Dict[str, any]:
    # 1) extraction
    text = utils.extract_text_safe(pdf_bytes)
    if not text or not text.strip():
        raise ValueError("Le PDF ne contient pas de texte exploitable.")
    max_chars = getattr(cfg, "SUMMARY_MAX_CHARS", 20000)
    if len(text) > max_chars:
        text = text[:max_chars]
    cleaned = " ".join(text.split())

    # 2) traduction
    english, _ = utils.maybe_translate(
        cleaned,
        target_lang=getattr(cfg, "TRANSLATION_TGT","en"),
        enable_offline=getattr(cfg, "ENABLE_OFFLINE_TRANSLATION",True),
    )

    # 3) build prefix from prefs
    prefs = prefs or {}
    prefix = _build_prefix(
        length=prefs.get("length","standard"),
        ton=prefs.get("ton","neutre"),
        focus=prefs.get("focus","général"),
        structure=prefs.get("structure","paragraphes"),
        couverture=prefs.get("couverture","concis"),
        style=prefs.get("style","abstractive"),
        chiffres=prefs.get("chiffres","garder"),
        citations=prefs.get("citations","inclure"),
    )

    # 4) load PT
    _ensure_pt()

    # 5) génération chunkée
    prompt = prefix + english
    max_tokens = getattr(cfg, "SUMMARY_PT_MAX_TOKENS",900)
    overlap    = getattr(cfg, "SUMMARY_PT_OVERLAP",150)
    gen_kwargs = dict(max_new_tokens=224, num_beams=2, early_stopping=True)
    summary_fr = utils.chunked_generate_text(
        _PT_MODEL, _PT_TOKENIZER, prompt,
        max_tokens=max_tokens, overlap=overlap,
        gen_kwargs=gen_kwargs
    )

    quality = utils.simple_quality_score(cleaned, summary_fr or "")

    return {
        "summary_fr": (summary_fr or "").strip(),
        "summary_en": "",
        "model_used": f"pt_mbart:{_PT_SOURCE}",
        "quality": float(quality),
    }
