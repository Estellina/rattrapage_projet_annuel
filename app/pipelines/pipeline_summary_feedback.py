from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EncoderNoRepeatNGramLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

from .pipeline_summary import (
    SumConfig,
    _ensure_en_model,
    _build_logits_processors,
    build_summary_prefix,
    translate_fr_to_en,
    translate_en_to_fr,
    clean_text_fr,
)

# =============================================================================
# Logging
# =============================================================================
log = logging.getLogger("summary_feedback_pipeline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# =============================================================================
# Protection / restauration des littéraux
# =============================================================================
REFS = r"\[\d+\]"
NUM = r"\b\d+(?:[.,]\d+)?\b"
UNITS_WORD = r"(?:MeV|GeV|K|Pa|bar|W|kW|MW|%|km|m|cm|mm|s|ms|µs)"
NUM_UNIT = rf"{NUM}\s*{UNITS_WORD}\b"
MPS = r"\b(?:\d+(?:[.,]\d+)?\s*)?(?:m/s|km/s)\b"
CHEM = r"(?:UF6|UO2|H2|O2|CO2|Xe|Ar)"
MATH = r"(?:∇|α|β|γ|θ|λ|μ|σ|Ω|≈|≃|≅|≤|≥|±)"

RX_REFS = re.compile(REFS)
RX_NUM = re.compile(NUM)
RX_NUMUNIT = re.compile(NUM_UNIT)
RX_MPS = re.compile(MPS)
RX_CHEM = re.compile(CHEM)
RX_MATH = re.compile(MATH)


def _normalize_placeholders(t: str) -> str:
    t = t.replace("&lt;", "<").replace("&gt;", ">")
    t = re.sub(r"<\s*([A-Za-z]{1,3}\d+)\s*>", r"<\1>", t)
    return t


def protect_numbers_and_symbols(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    out = text or ""
    patterns = [
        ("R", RX_REFS),
        ("U", RX_MPS),
        ("U", RX_NUMUNIT),
        ("C", RX_CHEM),
        ("M", RX_MATH),
        ("N", RX_NUM),
    ]
    counters = {k: 0 for k, _ in patterns}

    for key, rx in patterns:
        def _repl(m):
            tag = f"<{key}{counters[key]}>"; counters[key] += 1
            mapping[tag] = m.group(0)
            return tag
        out = rx.sub(_repl, out)
    return out, mapping


def restore_protected(text: str, mapping: Dict[str, str]) -> str:
    out = _normalize_placeholders(text or "")
    for tag, val in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(tag, val)
    return out


# =============================================================================
# Sélection de spans FR par intention
# =============================================================================
INTENT_METHODS = {"méthodes", "methodes", "methods", "methodology", "approche", "materials and methods"}
INTENT_DATA = {"données", "donnees", "data", "dataset", "materials"}

HEADERS_METHODS = re.compile(r"^(?:\d+(?:\.\d+)*)?\s*(?:méthod(?:e|ologie)s?|materials?\s+and\s+methods?|methods?|approach|procedure|experimental setup)\b", re.IGNORECASE)
HEADERS_DATA = re.compile(r"^(?:\d+(?:\.\d+)*)?\s*(?:données|donnees|data(?:set)?s?|materials?|corpus|échantillons|samples?)\b", re.IGNORECASE)


def _split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines()]


def _window(lines: List[str], start: int, radius: int = 8) -> str:
    lo = max(0, start); hi = min(len(lines), start + 1 + radius)
    return "\n".join(lines[lo:hi]).strip()


def _find_sections(lines: List[str], header_rx: re.Pattern) -> List[Tuple[int, str]]:
    return [(i, ln) for i, ln in enumerate(lines) if header_rx.search(ln)]


def _score_span_intent(span: str, intent: str, pos_weight: float) -> float:
    t = (span or "").lower()
    kws = [
        "méthode","méthodes","approche","procedure","protocole","pipeline","algorithme","architecture",
        "materials and methods","experimental setup","derivation","proof",
    ] if intent in INTENT_METHODS else [
        "données","dataset","datasets","corpus","materials","sources","samples","annotation","preprocessing",
        "splits","train","validation","test","k-fold","stratified",
    ]
    kw_hits = sum(1 for k in kws if k.lower() in t)
    kw_part = 0.5 * (kw_hits / max(1, len(kws)))

    num_hits = len(RX_NUM.findall(span)); unit_hits = len(RX_NUMUNIT.findall(span))
    chem_hits = len(RX_CHEM.findall(span)); ref_hits = len(RX_REFS.findall(span))
    math_hits = len(RX_MATH.findall(span)) if intent in INTENT_METHODS else 0
    val_part = 0.5 * min(1.0, (num_hits + unit_hits + chem_hits + ref_hits + math_hits) / 10.0)
    return kw_part + val_part + 0.1 * pos_weight


def select_topk_spans_by_intent_fr(text_fr: str, intent: str, *, k: int = 1, header_radius: int = 8, fallback_max_paragraphs: int = 8) -> List[str]:
    lines = _split_lines(text_fr)
    if not lines:
        return []
    header_rx = HEADERS_METHODS if intent in INTENT_METHODS else HEADERS_DATA if intent in INTENT_DATA else None
    candidates: List[Tuple[float, str]] = []
    if header_rx:
        for (i, _title) in _find_sections(lines, header_rx):
            span = _window(lines, i, radius=header_radius)
            score = _score_span_intent(span, intent, pos_weight=1.0)
            candidates.append((score, span))
    if not candidates:
        paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(lines)) if p.strip()]
        for p in paras[:fallback_max_paragraphs]:
            score = _score_span_intent(p, intent, pos_weight=0.2)
            candidates.append((score, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    K = max(1, min(k, len(candidates)))
    return [span for (_s, span) in candidates[:K]]


# =============================================================================
# Génération EN & pipeline
# =============================================================================
@torch.no_grad()
def _gen_one_en(model, tokenizer, prompt_en: str, cfg: SumConfig) -> str:
    enc = tokenizer(prompt_en, return_tensors="pt", truncation=True)
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


def build_feedback_prefix(selected: Dict[str, str]) -> str:
    if not selected:
        return build_summary_prefix()
    mapping = {"length": "length", "ton": "tone", "focus": "focus", "structure": "structure", "couverture": "coverage", "style": "style", "chiffres": "numbers", "citations": "citations"}
    parts = []
    for k, v in selected.items():
        kk = mapping.get(k, k); vv = (v or "").strip()
        if not vv or vv == "default":
            continue
        parts.append(f"<{kk}:{vv}>")
    return " ".join(parts) if parts else build_summary_prefix()


def run_summary_feedback_pipeline(pdf_bytes: bytes, prefs: Dict[str, str]) -> Dict[str, Any]:
    cfg = SumConfig()
    # 1) Texte FR source (extraction légère ici — utiliser utils.extract_text_safe côté app si besoin)
    from app import utils as _utils  # lazy import pour éviter cycles
    src_fr = _utils.extract_text_safe(pdf_bytes) or ""
    if not src_fr:
        return {"text_fr": "", "text_en": "", "model_used": "", "quality": 0.0}

    # 2) Sélection des spans par intention si fournie
    intent = (prefs or {}).get("focus", "")
    spans_fr = select_topk_spans_by_intent_fr(src_fr, intent) if intent else []
    spans_fr = spans_fr or [src_fr]

    # 3) Préfixe (seulement familles renseignées)
    prefix = build_feedback_prefix({k: v for k, v in (prefs or {}).items() if v and v != "default"})

    # 4) Modèle EN
    model, tokenizer, _dev, source = _ensure_en_model(cfg)

    # 5) Traduction FR→EN (spans) → génération EN
    spans_en = translate_fr_to_en(spans_fr, cfg)
    prompt_en = prefix + "\n\n" + "\n\n".join(spans_en)
    text_en = _gen_one_en(model, tokenizer, prompt_en, cfg)

    # 6) EN→FR et nettoyage
    text_fr = translate_en_to_fr(text_en, cfg)
    text_fr = clean_text_fr(text_fr)

    return {"text_fr": text_fr, "text_en": text_en, "model_used": source, "quality": 0.0}
