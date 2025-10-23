from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app import utils
from app.config import cfg
from app.models.pretrained_qg_flan import FlanT5QuestionGenerator, postprocess_questions
from app.pipelines.pipeline_summary import SumConfig as _SumConfig, translate_fr_to_en, translate_en_to_fr, clean_text_en
from app.pipelines.pipeline_questions import _ensure_qg_ready, _offline, _bf16_ok, _QG, _QG_SOURCE

# =============================================================================
# Logging
# =============================================================================
log = logging.getLogger("questions_feedback_pipeline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

# =============================================================================
# Helpers
# =============================================================================

def _map_payload_to_labels(payload: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    diff = (payload.get("difficulty") or "").lower()
    if diff in {"difficile", "hard", "advanced"}: labels.append("increase_difficulty")
    elif diff in {"facile", "easy", "beginner"}: labels.append("decrease_difficulty")
    scope = (payload.get("scope") or "").lower()
    if scope == "sections": labels.append("focus_section")
    if payload.get("avoid_trivial", False): labels.append("avoid_trivial")
    if payload.get("style"): labels.append("change_style")
    n = int(payload.get("n", 0) or 0)
    if n and n <= 3: labels.append("length_shorter")
    elif n and n >= 8: labels.append("length_longer")
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


# =============================================================================
# Sortie publique
# =============================================================================
@dataclass
class QuestionsOutput:
    questions_fr: List[str]
    questions_en: List[str]
    model_used: str


# =============================================================================
# Pipeline feedback
# =============================================================================

def run_questions_feedback_pipeline(
    pdf_bytes: bytes,
    num_questions: int = 5,
    payload: Optional[Dict[str, Any]] = None,
    questions_fr_seed: Optional[List[str]] = None,
) -> QuestionsOutput:
    payload = payload or {}

    # 1) Extraction FR
    raw_fr = utils.extract_text_safe(pdf_bytes)
    if not raw_fr:
        return QuestionsOutput(questions_fr=[], questions_en=[], model_used="")

    # 2) Traduction FR→EN
    cfg_sum = _SumConfig() if _SumConfig is not None else None
    full_en = clean_text_en("\n".join(translate_fr_to_en([raw_fr], cfg_sum or _SumConfig())))

    # 3) Chargement QG
    _ensure_qg_ready(); assert _QG is not None

    # 4) Mode réécriture vs régénération
    if questions_fr_seed:
        # réécriture : on traduit seed en EN, réécrit léger, puis retraduit FR
        seed_en = [clean_text_en(translate_fr_to_en([q], cfg_sum or _SumConfig())[0]) for q in questions_fr_seed]
        new_en = _QG.rewrite_questions(seed_en, labels=_map_payload_to_labels(payload))
        new_en = postprocess_questions(new_en)
        new_fr = [translate_en_to_fr(q, cfg_sum or _SumConfig()) for q in new_en]
        return QuestionsOutput(questions_fr=_dedupe_keep_order(new_fr), questions_en=_dedupe_keep_order(new_en), model_used=_QG_SOURCE)

    # régénération
    spans = re.split(r"(?<=[\.!?])\s+", full_en)
    passages = [" ".join(spans[i:i+1]).strip() for i in range(0, len(spans), 1)] or [full_en]
    qs_en = _QG.generate_from_spans(passages_en=passages, num_questions=max(1, int(num_questions)))
    qs_en = postprocess_questions(qs_en)
    qs_fr = [translate_en_to_fr(q, cfg_sum or _SumConfig()) for q in qs_en]
    return QuestionsOutput(questions_fr=_dedupe_keep_order(qs_fr), questions_en=_dedupe_keep_order(qs_en), model_used=_QG_SOURCE)
