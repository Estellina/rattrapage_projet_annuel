from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.models.fs_summary_loader import load_summary_model

import torch
from pypdf import PdfReader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EncoderNoRepeatNGramLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

# =============================================================================
# Logging
# =============================================================================
log = logging.getLogger("pipeline_summary")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# =============================================================================
# Config
# =============================================================================
@dataclass
class SumConfig:
    # Modèle EN (résumeur)
    hf_model_id: str = os.getenv("PT_SUMMARY_HF_ID", "facebook/bart-large-cnn")
    cache_dir: str = os.getenv(
        "PT_SUMMARY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache", "summary_en_model")
    )
    device: str = os.getenv("PT_DEVICE", "cuda")  # "cuda" ou "cpu"

    # Contexte/Chunking FR
    ctx_max: int = int(os.getenv("SUMMARY_PT_CTX_MAX", "1024"))
    safety_ratio: float = float(os.getenv("SUMMARY_CTX_SAFETY_RATIO", "0.9"))
    max_src_tokens: int = int(os.getenv("SUMMARY_PT_MAX_TOKENS", "900"))
    overlap_tokens: int = int(os.getenv("SUMMARY_PT_OVERLAP", "150"))

    # Top‑K (FR)
    topk_enable: bool = os.getenv("SUMMARY_TOPK_ENABLE", "true").lower() in {"1", "true", "yes"}
    topk_k: int = int(os.getenv("SUMMARY_TOPK_K", "1"))

    # Génération EN
    gen_max_new_tokens: int = int(os.getenv("SUMMARY_GEN_MAX_NEW_TOKENS", "224"))
    gen_num_beams: int = int(os.getenv("SUMMARY_GEN_NUM_BEAMS", "4"))
    gen_do_sample: bool = os.getenv("SUMMARY_GEN_DO_SAMPLE", "false").lower() in {"1", "true", "yes"}
    gen_length_penalty: float = float(os.getenv("SUMMARY_GEN_LENGTH_PENALTY", "1.05"))
    gen_early_stopping: bool = os.getenv("SUMMARY_GEN_EARLY_STOP", "false").lower() in {"1", "true", "yes"}
    no_repeat_ngram_size: int = int(os.getenv("SUMMARY_GEN_NO_REPEAT_NGRAM_SIZE", "4"))
    enc_no_repeat_ngram_size: int = int(os.getenv("SUMMARY_GEN_ENCODER_NO_REPEAT_NGRAM_SIZE", "3"))
    repetition_penalty: float = float(os.getenv("SUMMARY_GEN_REPETITION_PENALTY", "1.25"))
    min_new_tokens: int = int(os.getenv("SUMMARY_GEN_MIN_NEW_TOKENS", "60"))

    # Traduction (FR↔EN)
    trans_cache_dir: str = os.getenv(
        "TRANS_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache", "translators")
    )
    trans_device: str = os.getenv("TRANS_DEVICE", "cpu")
    trans_fr2en_id: str = os.getenv("TRANS_FR2EN_ID", "Helsinki-NLP/opus-mt-fr-en")
    trans_en2fr_id: str = os.getenv("TRANS_EN2FR_ID", "Helsinki-NLP/opus-mt-en-fr")

    # Divers
    joiner: str = "\n\n"
    max_chars_extract: int = int(os.getenv("SUMMARY_MAX_CHARS", "200000"))


# =============================================================================
# Singletons modèle & traducteurs
# =============================================================================
_SUM_MODEL: Optional[AutoModelForSeq2SeqLM] = None
_SUM_TOKENIZER: Optional[AutoTokenizer] = None
_SUM_DEVICE: Optional[torch.device] = None
_SUM_SOURCE: str = "unknown"

_FR2EN = None
_FR2EN_TOK = None
_EN2FR = None
_EN2FR_TOK = None
_TRANS_DEVICE: Optional[torch.device] = None


# =============================================================================
# Utils : extraction & nettoyage
# =============================================================================

def extract_text_from_pdf(pdf_bytes: bytes, max_chars: int) -> str:
    try:
        r = PdfReader(io.BytesIO(pdf_bytes))
        parts: List[str] = []
        for p in r.pages:
            t = p.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)[:max_chars]
    except Exception as e:
        log.error("PDF extract failed: %s", e)
        return ""


def clean_text_fr(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = re.sub(r"([!?.,;:])\1{1,}", r"\\1", t)
    t = re.sub(r"\s+([!?;:])", r" \1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    return t.strip()


def clean_text_en(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = t.replace("…", "...")
    t = re.sub(r"\.{3,}", "...", t)
    t = t.replace("...", "<ELL>")
    t = re.sub(r"([!?;,])\1{1,}", r"\\1", t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"\s+([!?.,;:])", r"\\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = t.replace("<ELL>", "...")
    return t.strip()


def detect_lang_simple(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "en"
    fr, en = 0, 0
    if any(c in t for c in "éèêàùûôîç"):
        fr += 2
    FR_SW = {"le","la","les","des","du","un","une","et","ou","de","dans","pour","avec","sur","par","au","aux","en","est","été","sont","que","qui"}
    EN_SW = {"the","and","or","of","in","to","for","with","on","by","is","are","was","were","that","which","who"}
    fr += sum(1 for w in FR_SW if f" {w} " in f" {t} ")
    en += sum(1 for w in EN_SW if f" {w} " in f" {t} ")
    if re.search(r"\b(l'|d'|qu'|j'|n')", t):
        fr += 1
    if re.search(r"\b(can't|don't|it's|you're|we're)\b", t):
        en += 1
    return "fr" if fr > en else "en"


# =============================================================================
# Chunking Top‑K (FR), puis traduction EN
# =============================================================================

def chunk_text_with_context(text: str, tokenizer, max_tokens: int, overlap: int) -> Tuple[List[List[int]], List[str]]:
    if not text or not text.strip():
        return [], []
    enc = tokenizer(text, add_special_tokens=False, return_tensors=None, return_attention_mask=False)
    ids = enc["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = [t for seq in ids for t in seq]
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap < 0 or overlap >= max_tokens:
        overlap = max(0, max_tokens // 6)
    step = max_tokens - overlap
    chunks_ids: List[List[int]] = []
    for start in range(0, len(ids), step):
        w = ids[start:start + max_tokens]
        if not w:
            break
        chunks_ids.append(w)
    chunks_txt = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    return chunks_ids, chunks_txt


FOCUS_KEYWORDS: Dict[str, List[str]] = {
    "méthodes": ["méthode", "method", "approche", "algorithme", "training", "architecture"],
    "détails": ["données", "dataset", "feature", "prétraitement", "hyperparam", "annotation"],
    "applications": ["application", "use case", "cas d'usage", "déploiement", "industrie", "production"],
    "résultats": ["résultat", "accuracy", "f1", "score", "benchmark", "table", "figure"],
    "limites": ["limite", "limitation", "future work", "contrainte", "biais", "risque", "erreur"],
}


def score_chunk_fr(text: str, focus: Optional[str]) -> float:
    t = (text or "").lower().strip()
    if not t:
        return 0.0
    if focus:
        kws = FOCUS_KEYWORDS.get(focus, [])
        kw_hits = sum(1 for k in kws if k in t)
        kw_part = 0.55 * (kw_hits / max(1, len(kws))) if kws else 0.0
    else:
        kw_part = 0.0
    digits = sum(c.isdigit() for c in t)
    num_part = 0.30 * min(digits / 50.0, 1.0)
    toks = t.split()
    uniq_ratio = len(set(toks)) / max(1, len(toks))
    div_part = 0.15 * uniq_ratio
    return float(kw_part + num_part + div_part)

def load_best_summary_model(cfg: SumConfig):
    """
    1) Essaie le modèle FS (from-scratch) si FS_SUMMARY_ENABLE=1.
    2) Sinon (ou si échec), utilise le loader historique `load_en_model`.
    Retourne: (model, tokenizer, device, source_tag)
    """
    try:
        if os.getenv("FS_SUMMARY_ENABLE", "false").lower() in {"1","true","yes"}:
            model, tok, device, src = load_summary_model(
                s3_prefix=os.getenv("FS_SUMMARY_S3_PREFIX", ""),
                cache_dir=os.getenv("FS_SUMMARY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache", "summary_en_model_fs")),
                device_pref=cfg.device_preference
            )
            return model, tok, device, f"fs:{src}"
    except Exception as e:
        log.warning(f"[SUMMARY][FS] fallback to pretrained ({e})")

    # fallback → pré-entraîné déjà existant
    m, t, d, used = load_summary_model(cfg)
    return m, t, d, used



def select_topk_chunks(
    text: str,
    tokenizer,
    *,
    k: int,
    focus: Optional[str],
    prefix: str,
    ctx_max: int,
    safety_ratio: float,
    max_tokens: int,
    overlap: int,
    enable_topk: bool,
) -> Dict[str, Any]:
    chunks_ids, chunks_txt = chunk_text_with_context(text, tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not chunks_txt:
        return dict(chunks_ids=[], chunks_txt=[], selected_idx=[], selected_txt=[], fast_path=False,
                    prefix_tokens=0, avg_chunk_tokens=0)
    prefix_tokens = len(tokenizer(prefix, add_special_tokens=False).input_ids) if prefix else 0
    avg_chunk_tokens = int(sum(len(tokenizer(c, add_special_tokens=False).input_ids) for c in chunks_txt) / max(1, len(chunks_txt)))
    fits_single = (prefix_tokens + avg_chunk_tokens) < int(safety_ratio * ctx_max)
    if not enable_topk:
        sel = list(range(len(chunks_txt))); fast = False
    else:
        scored = [(i, score_chunk_fr(chunks_txt[i], focus)) for i in range(len(chunks_txt))]
        scored.sort(key=lambda x: x[1], reverse=True)
        kk = max(1, min(k, len(scored)))
        if kk == 1 and fits_single:
            sel = [scored[0][0]]; fast = True
        else:
            sel = [i for (i, _) in scored[:kk]]; fast = False
    selected_txt = [chunks_txt[i] for i in sel]
    return dict(
        chunks_ids=chunks_ids,
        chunks_txt=chunks_txt,
        selected_idx=sel,
        selected_txt=selected_txt,
        fast_path=fast,
        prefix_tokens=int(prefix_tokens),
        avg_chunk_tokens=int(avg_chunk_tokens),
    )


# =============================================================================
# Traduction (Helsinki-NLP)
# =============================================================================

def _pick_device(name: str) -> torch.device:
    return torch.device("cuda" if (name == "cuda" and torch.cuda.is_available()) else "cpu")


def _load_translators(cfg: Optional[SumConfig] = None) -> None:
    global _FR2EN, _FR2EN_TOK, _EN2FR, _EN2FR_TOK, _TRANS_DEVICE
    if _FR2EN is not None and _EN2FR is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = cfg or SumConfig()
    cache_dir = Path(cfg.trans_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _TRANS_DEVICE = _pick_device(cfg.trans_device)

    _FR2EN_TOK = AutoTokenizer.from_pretrained(cfg.trans_fr2en_id, cache_dir=str(cache_dir), use_fast=True)
    _FR2EN = AutoModelForSeq2SeqLM.from_pretrained(cfg.trans_fr2en_id, cache_dir=str(cache_dir)).to(_TRANS_DEVICE).eval()
    _EN2FR_TOK = AutoTokenizer.from_pretrained(cfg.trans_en2fr_id, cache_dir=str(cache_dir), use_fast=True)
    _EN2FR = AutoModelForSeq2SeqLM.from_pretrained(cfg.trans_en2fr_id, cache_dir=str(cache_dir)).to(_TRANS_DEVICE).eval()


@torch.no_grad()
def translate_fr_to_en(texts: List[str], cfg: SumConfig, max_length: int = 480) -> List[str]:
    _load_translators(cfg)
    out: List[str] = []
    model_cap = int(getattr(_FR2EN.config, "max_position_embeddings", 512)) - 4
    safe_max = max(64, min(max_length, model_cap))
    for t in texts:
        if not t:
            out.append(""); continue
        enc = _FR2EN_TOK(t, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
        gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
        out.append(_FR2EN_TOK.decode(gen[0], skip_special_tokens=True).strip())
    return out


@torch.no_grad()
def translate_en_to_fr(text_en: str, cfg: SumConfig, max_length: int = 768) -> str:
    _load_translators(cfg)
    if not text_en:
        return ""
    enc = _EN2FR_TOK(text_en, return_tensors="pt", truncation=True, max_length=max_length).to(_TRANS_DEVICE)
    gen = _EN2FR.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
    return _EN2FR_TOK.decode(gen[0], skip_special_tokens=True).strip()


# =============================================================================
# Chargement modèle EN (résumeur)
# =============================================================================

def _ensure_en_model(cfg: Optional[SumConfig] = None) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device, str]:
    global _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE, _SUM_SOURCE
    if _SUM_MODEL is not None and _SUM_TOKENIZER is not None:
        return _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE or torch.device("cpu"), _SUM_SOURCE

    cfg = cfg or SumConfig()
    dev = _pick_device(cfg.device)
    tok = AutoTokenizer.from_pretrained(cfg.hf_model_id, cache_dir=cfg.cache_dir, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg.hf_model_id, cache_dir=cfg.cache_dir).to(dev).eval()

    _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE, _SUM_SOURCE = mdl, tok, dev, cfg.hf_model_id
    return mdl, tok, dev, _SUM_SOURCE


# =============================================================================
# Génération EN
# =============================================================================

def _build_logits_processors(tokenizer, min_len: int, rep_penalty: float, ngram: int, enc_input_ids=None):
    procs = LogitsProcessorList()
    if ngram > 0:
        procs.append(NoRepeatNGramLogitsProcessor(ngram))
    try:
        enc_ng = int(os.getenv("SUMMARY_GEN_ENCODER_NO_REPEAT_NGRAM_SIZE", "3"))
        if enc_input_ids is not None and enc_ng > 0:
            procs.append(EncoderNoRepeatNGramLogitsProcessor(enc_ng, enc_input_ids))
    except Exception:
        pass
    if rep_penalty and rep_penalty > 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(rep_penalty))
    if min_len and tokenizer.eos_token_id is not None:
        procs.append(MinLengthLogitsProcessor(min_len, tokenizer.eos_token_id))
    return procs


@torch.no_grad()
def _gen_one_en(model, tokenizer, prompt_en: str, cfg: SumConfig) -> str:
    enc = tokenizer(prompt_en, return_tensors="pt", truncation=True).to(_SUM_DEVICE or torch.device("cpu"))
    procs = _build_logits_processors(
        tokenizer,
        min_len=cfg.min_new_tokens,
        rep_penalty=cfg.repetition_penalty,
        ngram=cfg.no_repeat_ngram_size,
        enc_input_ids=enc.get("input_ids"),
    )
    gen = model.generate(
        **enc,
        max_new_tokens=cfg.gen_max_new_tokens,
        num_beams=cfg.gen_num_beams,
        do_sample=cfg.gen_do_sample,
        early_stopping=cfg.gen_early_stopping,
        length_penalty=cfg.gen_length_penalty,
        logits_processor=procs,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


# =============================================================================
# Préfixe de commande (contrôles)
# =============================================================================

def build_summary_prefix(selected: Optional[Dict[str, str]] = None) -> str:
    sel = selected or {}
    mapping = {
        "length": "length",
        "ton": "tone",
        "focus": "focus",
        "structure": "structure",
        "couverture": "coverage",
        "style": "style",
        "chiffres": "numbers",
        "citations": "citations",
    }
    parts: List[str] = []
    for k, v in sel.items():
        kk = mapping.get(k, k)
        vv = (v or "").strip()
        if not vv or vv == "default":
            continue
        parts.append(f"<{kk}:{vv}>")
    if not parts:
        return "<length:medium> <tone:neutral> <structure:paragraphs>"
    return " ".join(parts)


# =============================================================================
# Pipeline principale
# =============================================================================

def run_summary_pipeline(pdf_bytes: bytes, *, preferences: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    cfg = SumConfig()
    # 1) Extraction FR (ou EN)
    src_text = extract_text_from_pdf(pdf_bytes, max_chars=cfg.max_chars_extract)
    if not src_text:
        return {"text_fr": "", "text_en": "", "model_used": _SUM_SOURCE, "quality": 0.0}

    lang = detect_lang_simple(src_text)
    text_fr = src_text if lang == "fr" else src_text

    # 2) Top‑K (sur FR)
    model, tokenizer, device, used = load_best_summary_model(cfg)
    prefix = build_summary_prefix(preferences)
    chunks = select_topk_chunks(
        text_fr,
        tokenizer,
        k=cfg.topk_k,
        focus=(preferences or {}).get("focus"),
        prefix=prefix,
        ctx_max=cfg.ctx_max,
        safety_ratio=cfg.safety_ratio,
        max_tokens=cfg.max_src_tokens,
        overlap=cfg.overlap_tokens,
        enable_topk=cfg.topk_enable,
    )

    # 3) Traduction FR→EN des chunks
    en_chunks = translate_fr_to_en(chunks.get("selected_txt", []) or [text_fr], cfg)

    # 4) Génération EN (concat éventuellement)
    prompt_en = prefix + "\n\n" + cfg.joiner.join(en_chunks)
    text_en = _gen_one_en(model, tokenizer, prompt_en, cfg)

    # 5) Traduction EN→FR
    out_fr = translate_en_to_fr(text_en, cfg)

    # 6) Nettoyage/qualité simple
    out_fr = clean_text_fr(out_fr)
    quality = _simple_quality_score(src_text, out_fr)

    return {
        "text_fr": out_fr,
        "text_en": text_en,
        "quality": quality,
    }


# =============================================================================
# Qualité simple (diagnostic 0..1)
# =============================================================================

def _distinct_n(text: str, n: int = 2) -> float:
    t = " ".join((text or "").lower().split())
    toks = t.split()
    if len(toks) < n:
        return 1.0
    ngrams = set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return len(ngrams) / max(1, len(toks) - n + 1)


def _len_ratio(src: str, out: str, lo: float = 0.10, hi: float = 0.85) -> float:
    if not src or not out:
        return 0.0
    r = len(out) / max(1, len(src))
    if r < lo:
        return r / lo
    if r > hi:
        return max(0.0, 1.0 - (r - hi) / (2 * hi))
    return 1.0


def _overlap(src: str, out: str) -> float:
    if not src or not out:
        return 0.0
    S = set(src.lower().split()); O = set(out.lower().split())
    if not S:
        return 0.0
    return len(S & O) / max(1, len(S))


def _simple_quality_score(src_text: str, summary_text: str) -> float:
    src = (src_text or "").strip(); out = (summary_text or "").strip()
    if not out:
        return 0.0
    cov = _overlap(src, out)
    ratio = _len_ratio(src, out)
    d2 = _distinct_n(out, 2)
    d3 = _distinct_n(out, 3)
    punct = 1.0 - min(1.0, len(re.findall(r"[!?.,;:]{3,}", out)) / 3.0)
    score = 0.35 * cov + 0.25 * ratio + 0.20 * d2 + 0.10 * d3 + 0.10 * punct
    return round(max(0.0, min(1.0, score)), 3)
