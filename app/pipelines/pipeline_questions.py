from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import hashlib
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline

from app import utils
from app.config import cfg
from app.models.pretrained_qg_flan import (
    FlanT5QuestionGenerator,
    postprocess_questions,
)
from app.pipelines.pipeline_summary import (
    SumConfig as _SumConfig,
    translate_fr_to_en,
    translate_en_to_fr,
    clean_text_en,
)

# =============================================================================
# Logging
# =============================================================================
log = logging.getLogger("pipeline_questions")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

# =============================================================================
# ENV helpers
# =============================================================================

def _env_str(name: str, default: str = "") -> str:
    return str(getattr(cfg, name, os.getenv(name, default)) or "").strip()


def _offline() -> bool:
    return any(str(os.getenv(k, "0")).lower() in {"1", "true", "yes"} for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def _bf16_ok() -> bool:
    try:
        return torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    except Exception:
        return False

# =============================================================================
# Modèle QG (singleton)
# =============================================================================
_QG: Optional[FlanT5QuestionGenerator] = None
_QG_SOURCE: str = "unknown"


def _load_qg_from_hf(model_id: str):
    local_only = _offline()
    dev_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if _bf16_ok() else torch.float16
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype, device_map=dev_map, local_files_only=local_only).eval()
    return model, tok


def _ensure_qg_ready():
    global _QG, _QG_SOURCE
    if _QG is not None:
        return
    t0 = time.time()
    model_id = _env_str("PT_QG_MODEL", "google/flan-t5-large") or "google/flan-t5-large"
    model, tok = _load_qg_from_hf(model_id)
    qa = None
    try:
        if not _offline() and os.getenv("ENABLE_QA_FILTER", "0") in {"1", "true", "yes"}:
            dev = 0 if torch.cuda.is_available() else -1
            qa = hf_pipeline("question-answering", model="deepset/roberta-base-squad2", device=dev)
    except Exception as e:
        log.warning(f"[QG] QA filter not available: {e}")
    _QG = FlanT5QuestionGenerator(model, tok, qa_pipeline=qa, task_prefix="question: ")
    _QG_SOURCE = f"hf:{model_id}"
    log.info(f"[QG] ready from {_QG_SOURCE} in {time.time()-t0:.2f}s")

# =============================================================================
# Utils QG
# =============================================================================

def _split_into_spans(text_en: str, n_spans: int = 8, window: int = 1, max_span_len: int = 5) -> List[str]:
    # Découpe très simple en paragraphes/phrases, adaptable si besoin
    sents = re.split(r"(?<=[\.!?])\s+", (text_en or "").strip())
    if not sents:
        return []
    out: List[str] = []
    i = 0
    while i < len(sents) and len(out) < max(1, n_spans):
        w = " ".join(sents[i:i + max(1, window)])
        if w.strip():
            out.append(w.strip())
        i += max(1, window)
    return out


# =============================================================================
# Sortie publique
# =============================================================================
@dataclass
class QuestionsOutput:
    questions_fr: List[str]
    questions_en: List[str]
    model_used: str


# =============================================================================
# Pipeline principale
# =============================================================================

def run_questions_pipeline(pdf_bytes: bytes, *, num_questions: int = 5, doc_id: Optional[str] = None) -> QuestionsOutput:
    # 1) Extraction FR → EN
    cfg_sum = _SumConfig() if _SumConfig is not None else None
    raw_fr = utils.extract_text_safe(pdf_bytes)
    if not raw_fr:
        return QuestionsOutput(questions_fr=[], questions_en=[], model_used="")

    spans_en = translate_fr_to_en([raw_fr], cfg_sum or _SumConfig())
    full_text_en = clean_text_en("\n".join(spans_en))

    # 2) Chargement QG
    _ensure_qg_ready(); assert _QG is not None

    # 3) Spans & génération EN
    spans = _split_into_spans(full_text_en, n_spans=8, window=1)
    qs_en = _QG.generate_from_spans(passages_en=spans or [full_text_en], num_questions=max(1, int(num_questions)))
    qs_en = postprocess_questions(qs_en)

    # 4) Traduction EN→FR
    qs_fr = [translate_en_to_fr(q, cfg_sum or _SumConfig()) for q in qs_en]

    return QuestionsOutput(questions_fr=qs_fr, questions_en=qs_en, model_used=_QG_SOURCE)
