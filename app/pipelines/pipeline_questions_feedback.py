# app/pipelines/pipeline_questions_feedback.py
# -----------------------------------------------------------------------------
# Pipeline de FEEDBACK pour la génération de questions (QG)
# - lit le PDF (fourni par doc_id côté API)
# - traduit → EN si besoin
# - charge FLAN-T5 depuis S3 -> cache (ou HF fallback si offline désactivé)
# - applique le feedback : réécriture des questions existantes OU régénération
# - renvoie questions EN + FR
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import re
import time
import json
import logging
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

from app import utils
from app.config import cfg
from app.models.pretrained_qg_flan import FlanT5QuestionGenerator, postprocess_questions

# ───────────────────────────────── Logger
log = logging.getLogger("questions_feedback_pipeline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

try:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ───────────────────────────────── ENV helpers
def _env_str(name: str, default: str = "") -> str:
    return str(getattr(cfg, name, os.getenv(name, default)) or "").strip()

def _env_bool(name: str, default: bool = False) -> bool:
    v = str(getattr(cfg, name, os.getenv(name, str(default)))).strip().lower()
    return v in {"1", "true", "yes"}

def _env_int(name: str, default: int) -> int:
    try:
        return int(getattr(cfg, name, os.getenv(name, default)))
    except Exception:
        return default

def _offline() -> bool:
    return any(
        str(os.getenv(k, "0")).lower() in {"1", "true", "yes"}
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )

def _bf16_ok() -> bool:
    try:
        return torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    except Exception:
        return False

# ───────────────────────────────── S3 sync (copie de la pipeline Questions)
def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Bad S3 URI: {uri!r}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix

def _sync_s3_dir_to_local(s3_uri: str, dest_dir: str) -> bool:
    os.makedirs(dest_dir, exist_ok=True)
    # 1) boto3
    try:
        import boto3  # type: ignore
        bucket, prefix = _parse_s3_uri(s3_uri)
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)
        paginator = s3.get_paginator("list_objects_v2")
        any_obj = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):].lstrip("/")
                out_path = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                s3.download_file(bucket, key, out_path)
                any_obj = True
        if any_obj:
            return True
    except Exception as e:
        log.debug(f"[QG/FB][s3] boto3 sync failed: {e}")

    # 2) aws cli
    try:
        subprocess.run(["aws", "s3", "sync", s3_uri, dest_dir, "--no-progress"], check=True)
        return True
    except Exception as e:
        log.debug(f"[QG/FB][s3] aws cli sync failed: {e}")
        return False

def _ensure_local_model_dir() -> str:
    s3_prefix = _env_str("PT_QG_S3_PREFIX", "")
    if not s3_prefix:
        return ""
    cache_dir = _env_str("PT_QG_CACHE_DIR") or os.path.join(
        os.getenv("SA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache")),
        "qg_pt_flan",
    )
    ok = _sync_s3_dir_to_local(s3_prefix, cache_dir)

    def _exists(*names) -> bool:
        return any(os.path.exists(os.path.join(cache_dir, n)) for n in names)

    has_cfg = _exists("config.json")
    has_tok = _exists("tokenizer.json", "spiece.model")
    has_wts = _exists("pytorch_model.bin", "model.safetensors")
    if not (ok and has_cfg and has_tok and has_wts):
        msg = (f"S3 sync KO ou incomplet vers {cache_dir}. "
               f"Requis: config.json, tokenizer.json|spiece.model, pytorch_model.bin|model.safetensors.")
        if _env_bool("REQUIRE_S3", False):
            raise RuntimeError(msg + " (REQUIRE_S3=1)")
        log.warning("[QG/FB][s3] " + msg)
        return ""
    return cache_dir

# ───────────────────────────────── Chargement modèle/tokenizer/QA (singletons)
_QG: Optional[FlanT5QuestionGenerator] = None
_QG_SOURCE: str = "unknown"

def _load_qg_from_dir(local_dir: str):
    local_only = True
    dev_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if _bf16_ok() else torch.float16

    tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_dir, torch_dtype=dtype, device_map=dev_map, local_files_only=local_only
    ).eval()
    return model, tok

def _load_qg_from_hf(model_id: str):
    local_only = _offline()
    dev_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if _bf16_ok() else torch.float16

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=dev_map, local_files_only=local_only
    ).eval()
    return model, tok

def _ensure_qg_ready():
    global _QG, _QG_SOURCE
    if _QG is not None:
        return

    t0 = time.time()
    local_dir = ""
    try:
        local_dir = _ensure_local_model_dir()
    except Exception as e:
        log.error(f"[QG/FB] ensure S3 failed: {e}")
        local_dir = ""

    qa = None
    enable_qa = _env_bool("ENABLE_QA_FILTER", False) and not _offline()

    try:
        if local_dir:
            model, tok = _load_qg_from_dir(local_dir)
            _QG_SOURCE = f"s3:{_env_str('PT_QG_S3_PREFIX')}"
        else:
            model_id = _env_str("PT_QG_MODEL", "google/flan-t5-large") or "google/flan-t5-large"
            model, tok = _load_qg_from_hf(model_id)
            _QG_SOURCE = f"hf:{model_id}"
    except Exception as e:
        fb = _env_str("PT_QG_HF_MODEL", "google/flan-t5-large") or "google/flan-t5-large"
        log.warning(f"[QG/FB] primary load failed ({e}); fallback={fb}")
        model, tok = _load_qg_from_hf(fb)
        _QG_SOURCE = f"hf:{fb}"

    if enable_qa:
        try:
            dev = 0 if torch.cuda.is_available() else -1
            qa = hf_pipeline("question-answering", model="deepset/roberta-base-squad2", device=dev)
        except Exception as e:
            log.warning(f"[QG/FB] QA filter not available: {e}")

    _QG = FlanT5QuestionGenerator(model, tok, qa_pipeline=qa, task_prefix="question: ")
    log.info(f"[QG/FB] ready from {_QG_SOURCE} in {time.time()-t0:.2f}s")

# ───────────────────────────────── Aides feedback
def _map_payload_to_labels(payload: Dict[str, any]) -> List[str]:
    """
    Convertit le payload UI en labels 'rewrite' (augmenter/réduire difficulté, etc.)
    """
    labels: List[str] = []
    diff = (payload.get("difficulty") or "").lower()
    if diff in {"difficile", "hard", "advanced"}:
        labels.append("increase_difficulty")
    elif diff in {"facile", "easy", "beginner"}:
        labels.append("decrease_difficulty")

    scope = (payload.get("scope") or "").lower()
    if scope == "sections":
        labels.append("focus_section")

    if payload.get("avoid_trivial", False):
        labels.append("avoid_trivial")

    if payload.get("style"):
        labels.append("change_style")

    # longueur
    n = int(payload.get("n", 0) or 0)
    if n and n <= 3:
        labels.append("length_shorter")
    elif n and n >= 8:
        labels.append("length_longer")

    return labels

def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        t = (it or "").strip()
        if not t: continue
        key = "".join(ch for ch in t.lower() if ch.isalnum())
        if key in seen: continue
        seen.add(key); out.append(t)
    return out

def _split_questions_block(text: str) -> List[str]:
    if not text: return []
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("•*-–—").strip()
        if not line: continue
        parts = [p.strip() for p in line.split("?") if p.strip()]
        for p in parts:
            if not p.endswith("?"): p += "?"
            lines.append(p)
    return lines

# ───────────────────────────────── Sortie
@dataclass
class QuestionsOutput:
    questions_fr: List[str]
    questions_en: List[str]
    model_used: str

# ───────────────────────────────── Pipeline feedback
def run_questions_feedback_pipeline(
    pdf_bytes: bytes,
    num_questions: int = 5,
    payload: Optional[Dict[str, any]] = None,
    questions_fr_seed: Optional[List[str]] = None,
) -> QuestionsOutput:
    """
    - Si 'questions_fr_seed' est fourni : réécrit ces questions selon le feedback.
    - Sinon : régénère des questions en tenant compte du feedback (n, difficulté, etc.).
    """
    payload = payload or {}
    # 1) extraction
    t0 = time.time()
    raw = utils.extract_text_safe(pdf_bytes)
    if not raw or not raw.strip():
        raise ValueError("Le PDF ne contient pas de texte exploitable.")
    cleaned = " ".join(raw.split())
    MAX_CHARS = _env_int("QG_MAX_CHARS", 20000)
    if len(cleaned) > MAX_CHARS:
        cleaned = cleaned[:MAX_CHARS]
    log.info(f"[QG/FB][1/5 extract] {len(cleaned)} chars in {time.time()-t0:.2f}s")

    # 2) → EN
    t1 = time.time()
    english, _ = utils.maybe_translate(
        cleaned, target_lang="en",
        enable_offline=_env_bool("ENABLE_OFFLINE_TRANSLATION", True)
    )
    log.info(f"[QG/FB][2/5 translate→EN] in {time.time()-t1:.2f}s")

    # 3) modèle prêt
    _ensure_qg_ready()
    assert _QG is not None

    # 4) feedback → labels
    labels = _map_payload_to_labels(payload)

    # 5) chemin "rewrite" si seed fournie
    if questions_fr_seed and len(questions_fr_seed) > 0:
        # seed FR → EN
        seeds_en = []
        for q in questions_fr_seed:
            en, _ = utils.maybe_translate(
                q, target_lang="en",
                enable_offline=_env_bool("ENABLE_OFFLINE_TRANSLATION", True)
            )
            seeds_en.append(en)

        outs_en = _QG.rewrite_with_feedback(
            questions_en=seeds_en,
            source_text_en=english,
            labels=labels,
            payload=payload,
            max_new_tokens=_env_int("QG_GEN_MAX_NEW_TOKENS", 128)
        )

    else:
        # 6) chemin "regenerate"
        want = int(payload.get("n", num_questions) or num_questions)
        want = max(1, min(10, want))
        outs_en = _QG.generate(
            text_en=english,
            num_questions=want,
            per_span_ret=2,
            max_new_tokens=_env_int("QG_GEN_MAX_NEW_TOKENS", 128),
            use_qa_filter=_env_bool("ENABLE_QA_FILTER", False)
        )

    # 7) postprocess + FR
    outs_en = postprocess_questions(outs_en)
    outs_en = _dedupe_keep_order([q for q in outs_en if len(q) > 6])[:max(1, int(payload.get("n", num_questions) or num_questions))]

    outs_fr = [
        utils.maybe_translate(
            q, target_lang="fr",
            enable_offline=_env_bool("ENABLE_OFFLINE_TRANSLATION", True)
        )[0]
        for q in outs_en
    ]

    return QuestionsOutput(questions_fr=outs_fr, questions_en=outs_en, model_used=_QG_SOURCE)
