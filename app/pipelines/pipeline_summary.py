# app/pipelines/pipeline_summary.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.utils import (_score_chunk_en, _token_chunks_from_text)
import torch
from pypdf import PdfReader
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    EncoderNoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
    MarianMTModel,
    MarianTokenizer,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
@dataclass
class SumConfig:
    # English-only summarization model (finetuned)
    hf_model_id: str = os.getenv("PT_SUMMARY_HF_ID", "facebook/bart-large-cnn")
    s3_prefix: Optional[str] = os.getenv("PT_SUMMARY_S3_PREFIX")  # e.g., s3://bucket/checkpoint_bart
    cache_dir: str = os.getenv("PT_SUMMARY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache", "summary_en_model"))
    device_preference: str = os.getenv("PT_DEVICE", "cuda")  # "cuda" or "cpu"
    ignore_generation_config: bool = os.getenv("PT_IGNORE_GENCFG", "0") in {"1", "true", "yes"}

    # Translation models (Helsinki by default)
    trans_cache_dir: str = os.getenv("TRANS_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache", "translators"))
    trans_device: str = os.getenv("TRANS_DEVICE", "cpu")  # keep small & cheap
    trans_fr2en_id: str = os.getenv("TRANS_FR2EN_ID", "Helsinki-NLP/opus-mt-fr-en")
    trans_en2fr_id: str = os.getenv("TRANS_EN2FR_ID", "Helsinki-NLP/opus-mt-en-fr")

    # Chunking / context
    ctx_max: int = int(os.getenv("SUMMARY_PT_CTX_MAX", "1024"))
    safety_ratio: float = float(os.getenv("SUMMARY_CTX_SAFETY_RATIO", "0.8"))
    max_src_tokens: int = int(os.getenv("SUMMARY_PT_MAX_TOKENS", "900"))
    overlap_tokens: int = int(os.getenv("SUMMARY_PT_OVERLAP", "150"))

    # Top-K
    topk_enable: bool = os.getenv("SUMMARY_TOPK_ENABLE", "true").lower() in {"1", "true", "yes"}
    topk_k: int = int(os.getenv("SUMMARY_TOPK_K", "1"))

    # Generation
    gen_max_new_tokens: int = int(os.getenv("SUMMARY_GEN_MAX_NEW_TOKENS", "224"))
    gen_num_beams: int = int(os.getenv("SUMMARY_GEN_NUM_BEAMS", "4"))
    gen_do_sample: bool = os.getenv("SUMMARY_GEN_DO_SAMPLE", "false").lower() in {"1", "true", "yes"}
    gen_length_penalty: float = float(os.getenv("SUMMARY_GEN_LENGTH_PENALTY", "1.05"))
    gen_early_stopping: bool = os.getenv("SUMMARY_GEN_EARLY_STOP", "false").lower() in {"1", "true", "yes"}
    no_repeat_ngram_size: int = int(os.getenv("SUMMARY_GEN_NO_REPEAT_NGRAM_SIZE", "4"))
    enc_no_repeat_ngram_size: int = int(os.getenv("SUMMARY_GEN_ENCODER_NO_REPEAT_NGRAM_SIZE", "3"))
    repetition_penalty: float = float(os.getenv("SUMMARY_GEN_REPETITION_PENALTY", "1.25"))
    min_new_tokens: int = int(os.getenv("SUMMARY_GEN_MIN_NEW_TOKENS", "60"))

    # Misc
    joiner: str = "\n\n"
    max_chars_extract: int = int(os.getenv("SUMMARY_MAX_CHARS", "200000"))


# Singletons
_SUM_MODEL: Optional[AutoModelForSeq2SeqLM] = None
_SUM_TOKENIZER: Optional[AutoTokenizer] = None
_SUM_SOURCE: Optional[str] = None
_SUM_DEVICE: Optional[torch.device] = None

_FR2EN: Optional[MarianMTModel] = None
_FR2EN_TOK: Optional[MarianTokenizer] = None
_EN2FR: Optional[MarianMTModel] = None
_EN2FR_TOK: Optional[MarianTokenizer] = None
_TRANS_DEVICE: Optional[torch.device] = None


# ──────────────────────────────────────────────────────────────────────────────
def _find_model_dir(base: str) -> str:
    """
    Parcourt base et ses sous-dossiers pour trouver celui qui contient
    'config.json' + (pytorch_model.bin | model*.safetensors).
    Retourne base si pas mieux.
    """
    root = Path(base)
    if not root.exists():
        return base
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "config.json").exists() and ((p / "pytorch_model.bin").exists() or any(p.glob("model.safetensors"))):
            return str(p)
    return base

# Utils (extraction / nettoyage / qualité)
def extract_text_from_pdf(pdf_bytes: bytes, max_chars: int) -> str:
    try:
        r = PdfReader(io.BytesIO(pdf_bytes))
        chunks = []
        for page in r.pages:
            txt = page.extract_text() or ""
            chunks.append(txt)
        text = "\n".join(chunks)
        return text[:max_chars]
    except Exception as e:
        log.error("PDF extract failed: %s", e)
        return ""

def clean_text_fr(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = re.sub(r"([!?.,;:])\1{1,}", r"\1", t)
    t = re.sub(r"\s+([!?;:])", r" \1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    return t.strip()

import re

def clean_text_en(text: str) -> str:
    t = (text or "").strip()

    # Espace et retours à la ligne
    t = re.sub(r"[ \t]+", " ", t)          # compacter espaces/tabs
    t = re.sub(r"\s*\n\s*", "\n", t)       # nettoyer autour des sauts de ligne

    # Normaliser / préserver les ellipses
    t = t.replace("…", "...")              # Unicode ellipsis -> "..."
    t = re.sub(r"\.{3,}", "...", t)        # 4+ points -> "..."
    t = t.replace("...", "<ELLIPSIS>")     # placeholder temporaire

    # Réduire ponctuation répétée (hors ellipses)
    t = re.sub(r"([!?;,])\1{1,}", r"\1", t)  # "!!"->"!", "???"->"?"
    t = re.sub(r"\.{2,}", ".", t)            # ".." -> "."

    # Règles EN : pas d'espace AVANT la ponctuation
    t = re.sub(r"\s+([!?.,;:])", r"\1", t)

    # Parenthèses
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)

    # Restaurer les ellipses
    t = t.replace("<ELLIPSIS>", "...")

    return t.strip()


def dedupe_sentences_fr(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    seen = set()
    out = []
    for s in sents:
        k = s.strip().lower()
        if k and k not in seen:
            out.append(s.strip())
            seen.add(k)
    return " ".join(out).strip()

# Qualité simple (diagnostic 0..1)
def _distinct_n(text: str, n: int = 2) -> float:
    t = " ".join((text or "").lower().split())
    toks = t.split()
    if len(toks) < n: return 1.0
    ngrams = set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    return len(ngrams) / max(1, len(toks)-n+1)

def _len_ratio(src: str, out: str, lo: float = 0.10, hi: float = 0.85) -> float:
    if not src or not out: return 0.0
    r = len(out) / max(1, len(src))
    if r < lo: return r/lo
    if r > hi: return max(0.0, 1.0 - (r-hi)/(2*hi))
    return 1.0

def detect_lang_simple(text: str) -> str:
    """
    Heuristique rapide FR vs EN (pas de dépendance externe).
    Retourne 'fr' ou 'en'.
    """
    t = (text or "").lower()
    if not t:
        return "en"
    fr_hits = 0
    en_hits = 0
    # accents fréquents en FR
    if any(c in t for c in "éèêàùûôîç"):
        fr_hits += 2
    # stopwords simples
    FR_SW = {"le","la","les","des","du","un","une","et","ou","de","dans","pour","avec","sur","par","au","aux","en","est","été","sont","que","qui"}
    EN_SW = {"the","and","or","of","in","to","for","with","on","by","is","are","was","were","that","which","who"}
    fr_hits += sum(1 for w in FR_SW if f" {w} " in f" {t} ")
    en_hits += sum(1 for w in EN_SW if f" {w} " in f" {t} ")
    # formes typiques (articles/contr.)
    if re.search(r"\b(l'|d'|qu'|j'|n')", t): fr_hits += 1
    if re.search(r"\b(can't|don't|it's|you're|we're)\b", t): en_hits += 1
    return "fr" if fr_hits > en_hits else "en"


def _overlap(src: str, out: str) -> float:
    if not src or not out: return 0.0
    S = set(src.lower().split()); O = set(out.lower().split())
    if not S: return 0.0
    return len(S & O) / max(1, len(S))

def simple_quality_score(src_text: str, summary_text: str) -> float:
    src = (src_text or "").strip()
    out = (summary_text or "").strip()
    if not out: return 0.0
    cov = _overlap(src, out)
    ratio = _len_ratio(src, out)
    d2 = _distinct_n(out, 2)
    d3 = _distinct_n(out, 3)
    punct = 1.0 - min(1.0, len(re.findall(r"[!?.,;:]{3,}", out)) / 3.0)
    score = 0.35*cov + 0.25*ratio + 0.20*d2 + 0.10*d3 + 0.10*punct
    return round(max(0.0, min(1.0, score)), 3)


# ──────────────────────────────────────────────────────────────────────────────
# Top-K chunking (sur FR), puis on traduira ces chunks en EN
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

# ──────────────────────────────────────────────────────────────────────────────
# Translation loaders (Helsinki-NLP)
def _pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _load_translators(cfg: Optional[SumConfig] = None):
    import transformers
    global _FR2EN, _FR2EN_TOK, _EN2FR, _EN2FR_TOK, _TRANS_DEVICE
    if _FR2EN is not None and _EN2FR is not None:
        return

    class _Tmp: pass
    cfg = cfg or _Tmp()
    cache_dir = getattr(cfg, "trans_cache_dir", os.getenv("TRANS_CACHE_DIR")) or os.path.join(tempfile.gettempdir(), "sa_cache", "translators")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    fr2en_local = os.getenv("TRANS_FR2EN_LOCAL", "").strip()
    en2fr_local = os.getenv("TRANS_EN2FR_LOCAL", "").strip()
    fr2en_id = getattr(cfg, "trans_fr2en_id", os.getenv("TRANS_FR2EN_ID", "Helsinki-NLP/opus-mt-fr-en"))
    en2fr_id = getattr(cfg, "trans_en2fr_id", os.getenv("TRANS_EN2FR_ID", "Helsinki-NLP/opus-mt-en-fr"))
    local_only = any(os.getenv(k, "0").lower() in {"1","true","yes"} for k in ["HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","REQUIRE_LOCAL"])

    def _pick_path(pref: str, fallback_id: str, sub: str) -> str:
        if pref and os.path.isdir(pref) and os.path.exists(os.path.join(pref, "config.json")):
            return pref
        cand = os.path.join(cache_dir, sub)
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "config.json")):
            return cand
        return fallback_id

    fr2en_path = _pick_path(fr2en_local, fr2en_id, "fr-en")
    en2fr_path = _pick_path(en2fr_local, en2fr_id, "en-fr")

    _TRANS_DEVICE = torch.device(getattr(cfg, "trans_device", os.getenv("TRANS_DEVICE", "cpu")))
    _FR2EN_TOK = AutoTokenizer.from_pretrained(fr2en_path, cache_dir=cache_dir, use_fast=True,
                                               local_files_only=local_only)
    _FR2EN = AutoModelForSeq2SeqLM.from_pretrained(fr2en_path, cache_dir=cache_dir, local_files_only=local_only).to(
        _TRANS_DEVICE).eval()
    _EN2FR_TOK = AutoTokenizer.from_pretrained(en2fr_path, cache_dir=cache_dir, use_fast=True,
                                               local_files_only=local_only)
    _EN2FR = AutoModelForSeq2SeqLM.from_pretrained(en2fr_path, cache_dir=cache_dir, local_files_only=local_only).to(
        _TRANS_DEVICE).eval()
    log.info("[TRANS] FR->EN=%s | EN->FR=%s | local_only=%s", fr2en_path, en2fr_path, local_only)



@torch.no_grad()
def translate_fr_to_en(texts: List[str], cfg: SumConfig, max_length: int = 480) -> List[str]:
    """
    Traduit une liste de morceaux FR -> EN.
    Cappe la longueur au max_position_embeddings du modèle Marian (souvent 512).
    """
    _load_translators(cfg)
    out: List[str] = []
    # marge -4 pour les tokens spéciaux, évite "index out of range in self"
    safe_max = int(min(max_length, getattr(_FR2EN.config, "max_position_embeddings", 512) - 4))
    for t in texts:
        if not t:
            out.append("")
            continue
        enc = _FR2EN_TOK(t, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
        gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
        out.append(_FR2EN_TOK.decode(gen[0], skip_special_tokens=True).strip())
    return out



@torch.no_grad()
def translate_large_text_streaming_fr2en(text: str, cfg: SumConfig) -> str:
    _load_translators(cfg)
    if not text or not text.strip():
        return ""
    env_max = int(os.getenv("TRANS_MAX_LENGTH", str(getattr(cfg, "trans_max_length", 480))))
    model_cap = int(getattr(_FR2EN.config, "max_position_embeddings", 512)) - 4
    safe_max = max(64, min(env_max, model_cap))

    # split naïf en phrases; tu peux remplacer par spaCy si dispo
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    buf = ""
    outs = []
    for s in sents:
        cand = (buf + " " + s).strip() if buf else s
        # calcule len tokens; on utilise le tokenizer Marian
        enc_len = len(_FR2EN_TOK(cand, return_tensors="pt", truncation=True, max_length=safe_max)["input_ids"][0])
        if enc_len < safe_max:
            buf = cand
            continue
        if buf:
            enc = _FR2EN_TOK(buf, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
            gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
            outs.append(_FR2EN_TOK.decode(gen[0], skip_special_tokens=True))
        buf = s
    if buf:
        enc = _FR2EN_TOK(buf, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
        gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
        outs.append(_FR2EN_TOK.decode(gen[0], skip_special_tokens=True))
    return " ".join(outs)

@torch.no_grad()
def translate_en_to_fr(text: str, cfg: SumConfig) -> str:
    _load_translators(cfg)
    if not text or not text.strip():
        return ""
    env_max = int(os.getenv("TRANS_MAX_LENGTH", str(getattr(cfg, "trans_max_length", 480))))
    model_cap = int(getattr(_EN2FR.config, "max_position_embeddings", 512)) - 4
    safe_max = max(64, min(env_max, model_cap))
    enc = _EN2FR_TOK(text, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
    gen = _EN2FR.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
    return _EN2FR_TOK.decode(gen[0], skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (EN summarization model)
def _ensure_local_checkpoint_from_s3(prefix: str, cache_dir: str) -> Optional[str]:
    try:
        if not prefix or not prefix.startswith("s3://"):
            return None
        import boto3
        from urllib.parse import urlparse
        import re

        o = urlparse(prefix)
        bucket = o.netloc
        key = o.path.lstrip("/")
        if not bucket or not key:
            log.warning("S3 prefix mal formé: %s", prefix); return None

        local = Path(cache_dir)
        local.mkdir(parents=True, exist_ok=True)

        region = os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or None
        s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")

        allow = re.compile(
            r"(?:^|/)"
            r"(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|"
            r"special_tokens_map\.json|merges\.txt|vocab\.json|spm\.model|sentencepiece\.(?:bpe\.)?model|"
            r"pytorch_model\.bin|model(?:-\d+)?\.safetensors)$"
        )

        found = False
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            for obj in page.get("Contents", []) or []:
                k = obj["Key"]
                if not allow.search(k):
                    continue
                rel = k[len(key):].lstrip("/")
                if not rel:
                    continue
                dst = local / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and obj.get("Size") and dst.stat().st_size == obj["Size"]:
                    found = True
                    continue
                s3.download_file(bucket, k, str(dst))
                found = True

        if not found:
            log.warning("S3: aucun fichier d'inférence trouvé sous %s", prefix)
            return None

        model_dir = _find_model_dir(str(local))
        if not (Path(model_dir) / "config.json").exists():
            log.warning("S3: config.json introuvable dans %s", model_dir)
            return None
        return model_dir
    except Exception as e:
        import traceback
        log.warning("S3 download skipped (%s): %s\n%s", prefix, e, traceback.format_exc())
        return None



def _pick_sum_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_en_model(cfg: SumConfig) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device, str]:
    global _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE, _SUM_SOURCE
    if _SUM_MODEL is not None and _SUM_TOKENIZER is not None and _SUM_DEVICE is not None:
        return _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE, (_SUM_SOURCE or "unknown")

    cache_dir = cfg.cache_dir or os.path.join(tempfile.gettempdir(), "sa_cache", "summary_en_model")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # 1) LOCAL CACHE FIRST (répertoire déjà peuplé)
    local_model_dir = _find_model_dir(cache_dir)
    if (Path(local_model_dir) / "config.json").exists():
        log.info("[MODEL] Loading local checkpoint: %s", local_model_dir)
        _SUM_TOKENIZER = AutoTokenizer.from_pretrained(local_model_dir, cache_dir=cache_dir, use_fast=True, local_files_only=True)
        _SUM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(local_model_dir, cache_dir=cache_dir, local_files_only=True)
        used = local_model_dir
    else:
        # 2) S3 → cache
        used = ""
        if cfg.s3_prefix:
            s3_local = _ensure_local_checkpoint_from_s3(cfg.s3_prefix, cache_dir)
            if s3_local and (Path(s3_local) / "config.json").exists():
                log.info("[MODEL] Loading S3 checkpoint: %s", s3_local)
                _SUM_TOKENIZER = AutoTokenizer.from_pretrained(s3_local, cache_dir=cache_dir, use_fast=True, local_files_only=True)
                _SUM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(s3_local, cache_dir=cache_dir, local_files_only=True)
                used = s3_local

        # 3) Fallback HF si (1) et (2) indisponibles
        if not used:
            hf_id = os.getenv("PT_SUMMARY_HF_MODEL") or cfg.hf_model_id or os.getenv("PT_SUMMARY_HF_ID", "facebook/mbart-large-50")
            log.info("[MODEL] Loading HF model: %s", hf_id)
            _SUM_TOKENIZER = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir, use_fast=True)
            _SUM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(hf_id, cache_dir=cache_dir)
            used = hf_id

    _SUM_DEVICE = torch.device(cfg.device_preference if (cfg.device_preference in {"cuda", "cpu"} and (cfg.device_preference != "cuda" or torch.cuda.is_available())) else ("cuda" if torch.cuda.is_available() else "cpu"))
    _SUM_MODEL.to(_SUM_DEVICE).eval()
    _SUM_SOURCE = used
    return _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE, used


def _build_logits_processors(tokenizer, *, min_new_tokens: int, no_repeat_ngram_size: int,
                             encoder_no_repeat_ngram_size: int, repetition_penalty: float,
                             enc_input_ids: Optional[torch.Tensor]) -> LogitsProcessorList:
    procs = LogitsProcessorList()
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        procs.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size and encoder_no_repeat_ngram_size > 0 and enc_input_ids is not None:
        procs.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, enc_input_ids))
    if repetition_penalty and repetition_penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if min_new_tokens and tokenizer.eos_token_id is not None:
        procs.append(MinLengthLogitsProcessor(min_new_tokens, tokenizer.eos_token_id))
    return procs

# ──────────────────────────────────────────────────────────────────────────────
# Prefix de contrôle (langue agnostique : tokens spéciaux)
def build_summary_prefix() -> str:
    toks = ["<TONE_NEUTRAL>", "<LEN_MEDIUM>", "<STRUCT_PARAGRAPHS>", "<NUM_MINIMIZE>", "<CITE_EXCLUDE>","<DEDUP_ENFORCE>", "<COHESION_ENFORCE>", "<DOC_START>",]
    return " ".join([t for t in toks if t]) + " "

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
def summarize_text_raw(text_fr: str, cfg: Optional[SumConfig] = None, src_lang: str = "auto") -> Dict[str, Any]:
    cfg = cfg or SumConfig()
    model, tokenizer, device, used = load_en_model(cfg)
    _load_translators(cfg)

    # 1) Détection langue
    if src_lang == "auto":
        src_lang = detect_lang_simple(text_fr)

    # 2) Traduction entière en EN si FR (en flux sécurisé)
    t_en = text_fr
    route_head = "en"
    if src_lang == "fr":
        t_en = translate_large_text_streaming_fr2en(text_fr, cfg)
        route_head = "fr->en(trans)"

    # 3) Prefix contrôle
    prefix = build_summary_prefix()
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids

    # 4) Chunking EN + Top-K EN
    max_t = cfg.max_src_tokens
    overlap = cfg.overlap_tokens
    chunks_ids = _token_chunks_from_text(tokenizer, t_en, max_t, overlap)
    chunks_txt = [tokenizer.decode(x, skip_special_tokens=True) for x in chunks_ids]

    if cfg.topk_enable and len(chunks_txt) > 1:
        scored = [(i, _score_chunk_en(chunks_txt[i])) for i in range(len(chunks_txt))]
        scored.sort(key=lambda x: x[1], reverse=True)
        K = max(1, min(cfg.topk_k, len(scored)))
        selected_idx = [i for i, _ in scored[:K]]
    else:
        selected_idx = list(range(len(chunks_txt)))

    selected_en = [chunks_txt[i] for i in selected_idx]

    # 5) Génération EN (MAP) avec budget garanti
    ctx_max = cfg.ctx_max
    safety = cfg.safety_ratio
    budget = max(32, int(ctx_max * safety) - len(prefix_ids))

    def _gen_one_en(chunk_en: str) -> str:
        ids = tokenizer(chunk_en, add_special_tokens=False).input_ids
        if len(ids) > budget:
            ids = ids[:budget]
        input_ids = torch.tensor([prefix_ids + ids], device=device)
        procs = _build_logits_processors(
            tokenizer,
            min_new_tokens=cfg.min_new_tokens,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=cfg.enc_no_repeat_ngram_size,
            repetition_penalty=cfg.repetition_penalty,
            enc_input_ids=input_ids,
        )
        kwargs = dict(
            logits_processor=procs,
            max_new_tokens=cfg.gen_max_new_tokens,
            num_beams=cfg.gen_num_beams,
            do_sample=cfg.gen_do_sample,
            length_penalty=cfg.gen_length_penalty,
            early_stopping=cfg.gen_early_stopping,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        out = model.generate(input_ids=input_ids, **kwargs)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    partials_en: List[str] = []
    for i, ck_en in enumerate(selected_en):
        partials_en.append(_gen_one_en(ck_en))

    # 6) Reduce EN
    if len(partials_en) == 1:
        summary_en = partials_en[0]
    else:
        summary_en = cfg.joiner.join(partials_en)

    # 7) EN -> FR final + clean
    summary_fr = translate_en_to_fr(summary_en, cfg)
    summary_fr = clean_text_fr(dedupe_sentences_fr(summary_fr))

    route = f"{route_head}->sum_en->fr(trans) (topk={len(selected_idx)}) +{'map' if len(partials_en) > 1 else '1'}"
    quality = simple_quality_score(text_fr, summary_fr or "")

    return {
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "quality": float(quality),
        "model_used": used,
        "route": route,
        "lang_source": src_lang,
        "chunks_txt": chunks_txt,          # ici: chunks EN (stratégie english)
        "selected_chunks": selected_idx,
        "reuse_chunks_en": [chunks_txt[i] for i in selected_idx],  # prêt pour feedback
    }


def run_summary_pipeline(pdf_bytes: bytes) -> Dict[str, Any]:
    cfg = SumConfig()
    text = extract_text_from_pdf(pdf_bytes, max_chars=cfg.max_chars_extract)
    if not text or not text.strip():
        raise ValueError("Le PDF ne contient pas de texte exploitable.")
    lang = detect_lang_simple(text)
    if lang == "fr":
        text = clean_text_fr(text)
    else:
        text = clean_text_en(text)

    return summarize_text_raw(text, cfg)
