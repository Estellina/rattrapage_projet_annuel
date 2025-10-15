# app/pipelines/pipeline_summary_feedback.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from .pipeline_summary import (
    SumConfig,
    load_en_model,
    _load_translators,
    build_summary_prefix,                 # prefix générique
    _build_logits_processors,             # mêmes garde-fous génération
    translate_en_to_fr,                   # trad EN -> FR
    translate_fr_to_en,                   # trad FR -> EN (liste)
    extract_text_from_pdf,                # extraction PDF
    clean_text_fr,                        # nettoyage FR
    detect_lang_simple,                   # heuristique langue
 )

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EncoderNoRepeatNGramLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    MarianMTModel,
    MarianTokenizer,
)

from app import utils  # pour extract_text_safe si tu l'utilises ailleurs

log = logging.getLogger("summary_feedback_pipeline")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Nettoyage FR
def _clean_text_fr(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"([!?.,;:])\1{1,}", r"\1", t)
    return t

def _dedupe_sentences(text: str) -> str:
    if not text:
        return ""
    sents = re.split(r'(?<=[\.!?])\s+', text)
    seen, out = set(), []
    for s in sents:
        k = re.sub(r'\W+', '', s.lower())[:80]
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return " ".join(out)

# ──────────────────────────────────────────────────────────────────────────────
# Traduction (Helsinki-NLP)
_FR2EN: Optional[MarianMTModel] = None
_FR2EN_TOK: Optional[MarianTokenizer] = None
_EN2FR: Optional[MarianMTModel] = None
_EN2FR_TOK: Optional[MarianTokenizer] = None
_TRANS_DEVICE: Optional[torch.device] = None

def build_feedback_prefix(selected: Dict[str, str]) -> str:
    """
    Construit un prefix minimaliste à partir des familles VRAIMENT choisies.
    Si rien n'est choisi, on retombe sur build_summary_prefix().
    Exemple de rendu: "<length:short> <tone:neutral> <focus:methods>"
    Adapte la forme aux tokens attendus par ton modèle si besoin.
    """
    if not selected:
        return build_summary_prefix()

    parts = []
    # Harmonise les clés possiblement utilisées côté UI
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
    for k, v in selected.items():
        kk = mapping.get(k, k)
        vv = (v or "").strip()
        if not vv or vv == "default":
            continue
        parts.append(f"<{kk}:{vv}>")

    return " ".join(parts) if parts else build_summary_prefix()


# ──────────────────────────────────────────────────────────────────────────────
# Protection / restauration de littéraux (chiffres, unités, citations, chimie/simple)
_PROTECT_PATTERNS = [
    (r"\[\d+\]", "CITE"),  # citations [n]
    (r"\b\d+(?:[.,]\d+)?\s?(?:%|°C|K|Pa|kPa|MPa|m/s|km/s|MeV|GeV|ms|s|kg|g|mm|cm|m|μm|nm|mol|W|kW|MW|GHz|MHz|kHz|eV)\b", "NUMU"),
    (r"\b\d+(?:[.,]\d+)?\b", "NUM"),  # nombres nus (dernier recours)
    (r"\b(?:UF6|CO2|H2O|UO2)\b", "CHEM"),  # chimie simple (extensible)
]

def _protect_literals(text: str) -> tuple[str, dict[str, str]]:
    mapping = {}
    idx = 0
    out = text or ""
    for pat, tag in _PROTECT_PATTERNS:
        def _repl(m):
            nonlocal idx, mapping
            token = f"<<{tag}{idx}>>"
            mapping[token] = m.group(0)
            idx += 1
            return token
        out = re.sub(pat, _repl, out)
    return out, mapping

def _restore_literals(text: str, mapping: dict[str, str]) -> str:
    out = text or ""
    # remplace tokens plus longs d'abord pour éviter chevauchements
    for token in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(token, mapping[token])
    return out

# Wrappers de traduction qui préservent les littéraux
def fr2en_preserve_batch(texts: List[str], cfg: SumConfig) -> List[str]:
    protected, maps = [], []
    for t in texts:
        p, m = _protect_literals(t or "")
        protected.append(p); maps.append(m)
    en = translate_fr_to_en(protected, cfg)
    return [_restore_literals(e, maps[i]) for i, e in enumerate(en)]

def en2fr_preserve(text: str, cfg: SumConfig) -> str:
    p, m = _protect_literals(text or "")
    fr = translate_en_to_fr(p, cfg)
    return _restore_literals(fr, m)



def _pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Repérage de sections (titres) + scoring par intention

# --- REGEX & constantes ---
INTENT_METHODS = {"méthodes", "methodes", "methods", "methodology", "approche", "materials and methods"}
INTENT_DATA    = {"données", "donnees", "data", "dataset", "materials"}

HEADERS_METHODS = re.compile(
    r"^(?:\d+(?:\.\d+)*)?\s*(?:méthod(?:e|ologie)s?|materials?\s+and\s+methods?|methods?|approach|procedure|experimental setup)\b",
    re.IGNORECASE
)
HEADERS_DATA = re.compile(
    r"^(?:\d+(?:\.\d+)*)?\s*(?:données|donnees|data(?:set)?s?|materials?|corpus|échantillons|samples?)\b",
    re.IGNORECASE
)



# Références [12]
REFS  = r"\[\d+\]"

# Nombres (indépendants)
NUM   = r"\b\d+(?:[.,]\d+)?\b"

# Unités : ne matcher que si un NOMBRE est présent devant (ex: "42 Pa", "3.5 km")
# + cas spéciaux composés (m/s, km/s)
UNITS_WORD = r"(?:MeV|GeV|K|Pa|bar|W|kW|MW|%|km|m|cm|mm|s|ms|µs)"
NUM_UNIT   = rf"{NUM}\s*{UNITS_WORD}\b"
MPS        = r"\b(?:\d+(?:[.,]\d+)?\s*)?(?:m/s|km/s)\b"

# Chimie (exact, sensible à la casse)
CHEM  = r"(?:UF6|UO2|H2|O2|CO2|Xe|Ar)"

# Symboles/math (ok)
MATH  = r"(?:∇|α|β|γ|θ|λ|μ|σ|Ω|≈|≃|≅|≤|≥|±)"

RX_REFS   = re.compile(REFS)
RX_NUM    = re.compile(NUM)
RX_NUMUNIT= re.compile(NUM_UNIT)       # ex: "42 Pa", "3.5 km", "10 %"
RX_MPS    = re.compile(MPS)            # ex: "m/s", "12 m/s"
RX_CHEM   = re.compile(CHEM)
RX_MATH   = re.compile(MATH)


FOCUS_KWS_FOR_INTENT = {
    "méthodes": [
        "méthode","méthodes","approche","procedure","protocole","pipeline",
        "algorithme","architecture","implementation","materials and methods","experimental setup","derivation","proof"
    ],
    "données": [
        "données","dataset","datasets","corpus","materials","sources",
        "samples","annotation","preprocessing","splits","train","validation","test","k-fold","stratified"
    ],
}

def _normalize_placeholders(t: str) -> str:
    # Retire les éventuels échappements HTML et espaces dans les chevrons
    t = t.replace("&lt;", "<").replace("&gt;", ">")
    t = re.sub(r"<\s*([A-Za-z]{1,3}\d+)\s*>", r"<\1>", t)
    return t

def _split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines()]

def _window(lines: List[str], start: int, radius: int = 8) -> str:
    lo = max(0, start)
    hi = min(len(lines), start + 1 + radius)
    return "\n".join(lines[lo:hi]).strip()

def _find_sections(lines: List[str], header_rx: re.Pattern) -> List[Tuple[int, str]]:
    hits = []
    for i, ln in enumerate(lines):
        if header_rx.search(ln):
            hits.append((i, ln))
    return hits

def _score_span_intent(span: str, intent: str, pos_weight: float) -> float:
    t = (span or "").lower()
    kws = FOCUS_KWS_FOR_INTENT.get(
        "méthodes" if intent in INTENT_METHODS else "données" if intent in INTENT_DATA else "méthodes", []
    )
    kw_hits = sum(1 for k in kws if k.lower() in t)
    kw_part = 0.5 * (kw_hits / max(1, len(kws)))

    num_hits   = len(RX_NUM.findall(span))
    unit_hits  = len(RX_NUMUNIT.findall(span))
    chem_hits  = len(RX_CHEM.findall(span))
    ref_hits   = len(RX_REFS.findall(span))
    math_hits  = len(RX_MATH.findall(span)) if intent in INTENT_METHODS else 0
    val_part = 0.5 * min(1.0, (num_hits + unit_hits + chem_hits + ref_hits + math_hits) / 10.0)

    return kw_part + val_part + 0.1 * pos_weight

def select_topk_spans_by_intent_fr(
    text_fr: str,
    intent: str,
    *,
    k: int = 1,
    header_radius: int = 8,
    fallback_max_paragraphs: int = 8
) -> List[str]:
    lines = _split_lines(text_fr)
    if not lines:
        return []

    header_rx = HEADERS_METHODS if intent in INTENT_METHODS else HEADERS_DATA if intent in INTENT_DATA else None
    candidates: List[Tuple[float, str]] = []

    if header_rx:
        hits = _find_sections(lines, header_rx)
        for (i, _title) in hits:
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

def protect_numbers_and_symbols(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Remplace références, m/s|km/s, nombres+unités, chimie, math, nombres simples par des tags.
    Ordre important: on protège d'abord ce qui est le plus structurel.
    """
    mapping: Dict[str, str] = {}
    out = text

    patterns = [
        ("R", RX_REFS),     # [12]
        ("U", RX_MPS),      # m/s, km/s
        ("U", RX_NUMUNIT),  # 42 Pa, 3.5 km, 10 %
        ("C", RX_CHEM),     # UF6
        ("M", RX_MATH),     # α, ∇, …
        ("N", RX_NUM),      # 99.75  (après num+unit pour éviter le double match)
    ]
    counters = {"R":0, "U":0, "C":0, "M":0, "N":0}

    for key, rx in patterns:
        def _repl(m):
            tag = f"<{key}{counters[key]}>"
            counters[key] += 1
            mapping[tag] = m.group(0)
            return tag
        out = rx.sub(_repl, out)

    return out, mapping

def restore_protected(text: str, mapping: Dict[str, str]) -> str:
    out = _normalize_placeholders(text)
    # Remplacement du plus long tag au plus court (sécurité)
    for tag, val in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(tag, val)
    return out


def restore_protected(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for tag, val in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(tag, val)
    return out

@torch.no_grad()
def translate_fr_to_en_protected(texts_fr: List[str], cfg, max_length: int = 480) -> List[str]:
    _load_translators()
    outs = []
    for t in texts_fr:
        prot, mp = protect_numbers_and_symbols(t)
        enc = _FR2EN_TOK(prot, return_tensors="pt", truncation=True, max_length=max_length).to(_TRANS_DEVICE)
        gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
        en = _FR2EN_TOK.decode(gen[0], skip_special_tokens=True).strip()
        en = _normalize_placeholders(en)
        outs.append(restore_protected(en, mp))
    return outs

@torch.no_grad()
def translate_en_to_fr_protected(text_en: str, cfg, max_length: int = 480) -> str:
    _load_translators()
    prot, mp = protect_numbers_and_symbols(text_en)
    enc = _EN2FR_TOK(prot, return_tensors="pt", truncation=True, max_length=max_length).to(_TRANS_DEVICE)
    gen = _EN2FR.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
    fr = _EN2FR_TOK.decode(gen[0], skip_special_tokens=True).strip()
    fr = _normalize_placeholders(fr)
    return restore_protected(fr, mp)




def _load_translators():
    global _FR2EN, _FR2EN_TOK, _EN2FR, _EN2FR_TOK, _TRANS_DEVICE
    if _FR2EN and _EN2FR and _TRANS_DEVICE:
        return
    cache_dir = os.getenv("TRANS_CACHE_DIR", os.path.join(Path(os.getenv("TMPDIR", "/tmp")), "sa_cache", "translators"))
    _TRANS_DEVICE = _pick_device(os.getenv("TRANS_DEVICE", "cpu"))
    fr2en_id = os.getenv("TRANS_FR2EN_ID", "Helsinki-NLP/opus-mt-fr-en")
    en2fr_id = os.getenv("TRANS_EN2FR_ID", "Helsinki-NLP/opus-mt-en-fr")
    _FR2EN_TOK = MarianTokenizer.from_pretrained(fr2en_id, cache_dir=cache_dir)
    _FR2EN = MarianMTModel.from_pretrained(fr2en_id, cache_dir=cache_dir).to(_TRANS_DEVICE).eval()
    _EN2FR_TOK = MarianTokenizer.from_pretrained(en2fr_id, cache_dir=cache_dir)
    _EN2FR = MarianMTModel.from_pretrained(en2fr_id, cache_dir=cache_dir).to(_TRANS_DEVICE).eval()

@torch.no_grad()
def _fr2en_batch(chunks_fr: List[str]) -> List[str]:
    """
    Cap Marian (≈512) avec marge de sécurité, évite les IndexError.
    """
    _load_translators()
    if not chunks_fr:
        return []

    env_max = int(os.getenv("TRANS_MAX_LENGTH", "480"))
    model_cap = int(getattr(_FR2EN.config, "max_position_embeddings", 512)) - 4
    safe_max = max(64, min(env_max, model_cap))

    outs: List[str] = []
    for t in chunks_fr:
        txt = (t or "").strip()
        if not txt:
            outs.append("")
            continue
        enc = _FR2EN_TOK(txt, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
        gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
        outs.append(_FR2EN_TOK.decode(gen[0], skip_special_tokens=True).strip())
    return outs


def _budget_for_chunk(tokenizer, prefix: str) -> Tuple[List[int], int]:
    """
    Calcule le budget de tokens pour BART/MBART (ctx_max * safety - len(prefix)).
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    ctx_max = int(os.getenv("SUMMARY_PT_CTX_MAX", "1024"))
    safety = float(os.getenv("SUMMARY_CTX_SAFETY_RATIO", "0.90"))
    budget = max(32, int(ctx_max * safety) - len(prefix_ids))
    return prefix_ids, budget

@torch.no_grad()
@torch.no_grad()
def _fr2en_batch(chunks_fr: List[str]) -> List[str]:
    _load_translators()
    if not chunks_fr:
        return []
    env_max = int(os.getenv("TRANS_MAX_LENGTH", "480"))
    model_cap = int(getattr(_FR2EN.config, "max_position_embeddings", 512)) - 4
    safe_max = max(64, min(env_max, model_cap))
    outs: List[str] = []
    for t in chunks_fr:
        txt = (t or "").strip()
        if not txt:
            outs.append("")
            continue
        enc = _FR2EN_TOK(txt, return_tensors="pt", truncation=True, max_length=safe_max).to(_TRANS_DEVICE)
        gen = _FR2EN.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
        outs.append(_FR2EN_TOK.decode(gen[0], skip_special_tokens=True).strip())
    return outs

def _gen_budget(tokenizer, prefix: str) -> Tuple[List[int], int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    ctx_max = int(os.getenv("SUMMARY_PT_CTX_MAX", "1024"))
    safety = float(os.getenv("SUMMARY_CTX_SAFETY_RATIO", "0.90"))
    budget = max(32, int(ctx_max * safety) - len(prefix_ids))
    return prefix_ids, budget


@torch.no_grad()
def _en2fr(text: str, max_length: int = 768) -> str:
    _load_translators()
    if not text: return ""
    enc = _EN2FR_TOK(text, return_tensors="pt", truncation=True, max_length=max_length).to(_TRANS_DEVICE)
    gen = _EN2FR.generate(**enc, max_new_tokens=256, num_beams=4, early_stopping=True)
    return _EN2FR_TOK.decode(gen[0], skip_special_tokens=True)

# ──────────────────────────────────────────────────────────────────────────────
# Modèle EN (même que summary)
_SUM_MODEL: Optional[AutoModelForSeq2SeqLM] = None
_SUM_TOKENIZER: Optional[AutoTokenizer] = None
_SUM_DEVICE: Optional[torch.device] = None
_SUM_SOURCE: str = "unknown"

def _ensure_sum_model():
    global _SUM_MODEL, _SUM_TOKENIZER, _SUM_DEVICE, _SUM_SOURCE
    if _SUM_MODEL and _SUM_TOKENIZER and _SUM_DEVICE:
        return
    cache_dir = os.getenv("PT_SUMMARY_CACHE_DIR")
    model_id = os.getenv("PT_SUMMARY_HF_ID", "facebook/bart-large-cnn")
    _SUM_TOKENIZER = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
    _SUM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
    _SUM_DEVICE = torch.device("cuda" if (os.getenv("PT_DEVICE", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")
    _SUM_MODEL.to(_SUM_DEVICE).eval()
    _SUM_SOURCE = model_id

# ──────────────────────────────────────────────────────────────────────────────
# Logits processors / Génération EN
def _legacy_build_logits_processors(tokenizer, min_len: int, rep_penalty: float, ngram: int, enc_input_ids=None):
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
def _gen_one_en(model, tokenizer, prompt_en: str) -> str:
    enc = tokenizer(prompt_en, return_tensors="pt", truncation=True)
    enc = {k: v.to(_SUM_MODEL.device) for k, v in enc.items()}
    procs = _legacy_build_logits_processors(tokenizer,
                                     min_len=int(os.getenv("SUMMARY_GEN_MIN_NEW_TOKENS", "60")),
                                     rep_penalty=float(os.getenv("SUMMARY_GEN_REPETITION_PENALTY", "1.25")),
                                     ngram=int(os.getenv("SUMMARY_GEN_NO_REPEAT_NGRAM_SIZE", "4")),
                                     enc_input_ids=enc.get("input_ids"))
    gen_kwargs = dict(
        max_new_tokens=int(os.getenv("SUMMARY_GEN_MAX_NEW_TOKENS", "224")),
        num_beams=int(os.getenv("SUMMARY_GEN_NUM_BEAMS", "4")),
        do_sample=os.getenv("SUMMARY_GEN_DO_SAMPLE", "false").lower() == "true",
        logits_processor=procs,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    out = model.generate(**enc, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Prefix (mêmes tokens spéciaux, agnostiques langue)
def _build_prefix_optional(prefs: Dict[str, str]) -> str:
    LENGTH2TOKEN = {"court": "<LEN_SHORT>", "standard": "<LEN_MEDIUM>", "long": "<LEN_LONG>"}
    TONE2TOKEN   = {"formel": "<TONE_FORMAL>", "neutre": "<TONE_NEUTRAL>", "informel": "<TONE_CASUAL>"}
    FOCUS2TOKEN  = {
        "général": "<FOCUS_GENERAL>", "détails": "<FOCUS_DETAILS>", "résultats": "<FOCUS_RESULTS>",
        "méthodes": "<FOCUS_METHODS>", "limites": "<FOCUS_LIMITATIONS>", "applications": "<FOCUS_APPLICATIONS>"
    }
    STRUCT2TOKEN = {"paragraphes": "<STRUCT_PARAGRAPHS>", "puces": "<STRUCT_BULLETS>", "sections": "<STRUCT_SECTIONS>"}
    COVER2TOKEN  = {"concis": "<COVER_KEYPOINTS>", "complet": "<COVER_COMPREHENSIVE>"}
    STYLE2TOKEN  = {"abstractive": "<STYLE_ABSTRACTIVE>", "extractive": "<STYLE_EXTRACTIVE>"}
    NUM2TOKEN    = {"garder": "<NUM_KEEP>", "réduire": "<NUM_MINIMIZE>"}
    CITE2TOKEN   = {"inclure": "<CITE_INCLUDE>", "exclure": "<CITE_EXCLUDE>"}

    toks: List[str] = []
    if "length"     in prefs: toks.append(LENGTH2TOKEN.get(prefs["length"], ""))
    if "ton"        in prefs: toks.append(TONE2TOKEN.get(prefs["ton"], ""))
    if "focus"      in prefs: toks.append(FOCUS2TOKEN.get(prefs["focus"], ""))
    if "structure"  in prefs: toks.append(STRUCT2TOKEN.get(prefs["structure"], ""))
    if "couverture" in prefs: toks.append(COVER2TOKEN.get(prefs["couverture"], ""))
    if "style"      in prefs: toks.append(STYLE2TOKEN.get(prefs["style"], ""))
    if "chiffres"   in prefs: toks.append(NUM2TOKEN.get(prefs["chiffres"], ""))
    if "citations"  in prefs: toks.append(CITE2TOKEN.get(prefs["citations"], ""))

    toks = [t for t in toks if t]
    toks.append("<DOC_START>")
    return " ".join(toks) + " "

# ──────────────────────────────────────────────────────────────────────────────
# Scoring FR (si on doit re-TopK)
_FOCUS_KEYWORDS: Dict[str, List[str]] = {
    # Sections / rhétorique scientifique (FR/EN)
    "introduction": [
        "introduction", "contexte", "background", "motivation", "objectif",
        "problem statement", "overview"
    ],
    "méthodes": [
        "méthode", "méthodes", "method", "methods", "methodology",
        "approche", "approach", "procédure", "procedure",
        "protocole", "pipeline", "workflow",
        "algorithme", "algorithm", "heuristic", "derivation", "proof",
        "architecture", "implementation", "experimental setup",
        "materials and methods", "reproduction", "reproducibility"
    ],
    "données": [
        "donnée", "données", "data", "dataset", "datasets", "corpus",
        "material", "materials", "sources", "échantillon", "échantillons", "samples",
        "collection", "acquisition", "annotation", "labelling", "ground truth",
        "prétraitement", "preprocessing", "nettoyage", "cleaning",
        "split", "train", "validation", "test", "hold-out", "k-fold", "stratified",
        "metadonnées", "metadata", "biais", "bias", "leakage", "privacy", "consent"
    ],
    "résultats": [
        "résultat", "résultats", "results", "findings", "observation",
        "accuracy", "precision", "recall", "f1", "auc", "ap",
        "benchmark", "baselines", "state of the art", "sota",
        "ablation", "sensitivity analysis",
        "table", "figure", "box plot", "confidence interval", "95% ci"
    ],
    "discussion": [
        "discussion", "analysis", "interpretation", "insight", "qualitative"
    ],
    "limites": [
        "limite", "limitations", "contrainte", "constraint",
        "risk", "risque", "biais", "bias", "erreur", "error",
        "threats to validity", "menace à la validité", "validité interne",
        "future work", "perspectives"
    ],
    "applications": [
        "application", "applications", "use case", "use cases",
        "cas d'usage", "déploiement", "deployment", "industrie",
        "production", "real-world", "inference time", "latency", "throughput", "cost"
    ],

    # Domaine 1 : plasma / fusion / propulsion
    "plasma": [
        "force-free", "woltjer", "taylor", "grad-shafranov",
        "helicity", "poloidal", "toroidal", "spheromak",
        "bessel", "j0", "j1", "eigenmode", "boundary conditions",
        "magnetic confinement", "tokamak", "instabilities", "gold-hoyle",
        "uf6", "uranium hexafluoride", "fission fragments", "opacity",
        "ideal gas law", "particle density", "temperature", "pressure",
        "specific impulse", "thermal stability", "mev", "m/s", "k", "pa"
    ],

    # Domaine 2 : sécurité PDF / malware
    "pdf_malware": [
        "pdf structure", "header", "body", "xref", "cross-reference", "trailer",
        "object", "objects", "indirect object", "stream", "endstream", "/length",
        "/filter", "/root", "/size", "object stream", "objstm", "jbig2",
        "acroform", "/aa", "openaction", "launch", "/js", "javascript", "uri",
        "embedded file", "file attachment", "annotation", "action",
        "graph features", "tree", "num nodes", "num leaves", "num edges",
        "avg children", "median children", "depth", "density", "assortativity",
        "avg shortest path", "avg clustering coefficient", "variance children",
        "random forest", "decision tree", "svm", "mlp", "xgboost",
        "cross-validation", "confusion matrix", "tpr", "fpr", "fnr", "tnr",
        "contagio", "benign", "malicious", "evasion", "adversarial",
        "99.75% accuracy", "95% ci", "box plot", "pdfrw", "poppler", "hidost", "pdfrate"
    ],

    # Aides au repérage (équations, unités, références)
    "equations": [
        "equation", "équation", "[equation]", "∇", "α", "β", "γ", "λ", "μ", "σ", "≈", "≤", "≥"
    ],
    "unités": [
        "mev", "gev", "m/s", "km/s", "k", "pa", "bar", "w", "kw", "mw", "%",
        "sec", "s", "ms", "µs", "km", "m", "cm", "mm", "objects", "nodes"
    ],
    "citations": [
        "[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]"
    ],
}

def _normalize_intent(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"méthode", "method"}: return "méthodes"
    if s in {"donnee"}: return "données"
    if s in {"general"}: return "général"
    return s


def _score_chunk_fr(text: str, focus: Optional[str]) -> float:
    text_l = (text or "").lower().strip()
    if not text_l:
        return 0.0
    if focus:
        kws = _FOCUS_KEYWORDS.get(focus, [])
        kw_hits = sum(1 for k in kws if k in text_l)
        kw_part = 0.55 * (kw_hits / max(1, len(kws)))
    else:
        kw_part = 0.0
    digits = sum(c.isdigit() for c in text_l)
    uniq_ratio = len(set(text_l.split())) / max(1, len(text_l.split()))
    return kw_part + 0.30 * min(digits / 50.0, 1.0) + 0.15 * uniq_ratio

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline feedback (FR→EN→sum EN→EN→FR)

@torch.no_grad()
def run_summary_feedback_pipeline(pdf_bytes: bytes, prefs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Feedback pipeline 'intention-aware' :
    1) Extrait le texte FR du PDF.
    2) Sélectionne Top-K passages selon l'intention (méthodes/données) via en-têtes + valeurs clés.
    3) Traduit FR->EN en protégeant chiffres/symboles/références/unités.
    4) Concatène et résume en EN (modèle EN).
    5) Traduit EN->FR (protégé).
    6) Retourne FR, EN, les spans FR utilisés et les métadonnées.
    """
    t0 = time.time()
    prefs = prefs or {}
    cfg = SumConfig()

    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Aucun PDF reçu : fournissez un doc_id valide.")

    # 1) Extraction & nettoyage FR
    raw_fr = extract_text_from_pdf(pdf_bytes, max_chars=getattr(cfg, "max_chars_extract", 400000))
    if not raw_fr:
        raise ValueError("PDF vide ou non lisible.")
    try:
        raw_fr = clean_text_fr(raw_fr)
    except Exception:
        raw_fr = (raw_fr or "").strip()

    # 2) Intention & K
    intent = _normalize_intent(prefs.get("focus") or "général")
    topk_k = int(prefs.get("topk_k") or os.getenv("FEEDBACK_TOPK_K", "2"))

    # 3) Sélection Top-K (FR)
    if intent in INTENT_METHODS or intent in INTENT_DATA:
        spans_fr = select_topk_spans_by_intent_fr(raw_fr, intent, k=topk_k)
    else:
        spans_fr = select_topk_spans_by_intent_fr(raw_fr, "méthodes", k=topk_k)
    if not spans_fr:
        raise ValueError("Impossible de sélectionner des passages pertinents pour le feedback.")

    # 4) FR -> EN protégé
    trans_max_len = int(os.getenv("TRANS_MAX_LENGTH", "480"))
    chunks_en = translate_fr_to_en_protected(spans_fr, cfg, max_length=trans_max_len)


    # 5) Modèle EN (même chargeur que la pipeline summary)
    model_en, tok_en, device, used = load_en_model(cfg)

    # 6) Prefix optionnel depuis prefs
    prefix_en = _build_prefix_optional(prefs)

    # 7) Prompt EN (court + consignes)
    if intent in INTENT_METHODS:
        focus_line = "Focus on METHODS: describe algorithms/derivations/setup concisely; keep citations like [n]; keep numbers/units."
    elif intent in INTENT_DATA:
        focus_line = "Focus on DATA: sources, size, structure, preprocessing, splits; keep citations like [n]; keep numbers/units."
    else:
        focus_line = "Focus on key points; keep citations like [n]; keep numbers/units."

    joined_en = prefix_en + focus_line + "\n\n" + "\n\n---\n\n".join(chunks_en)

    # 8) Encodage + génération EN
    # 8) Encodage + génération EN
    enc = tok_en(
        joined_en,
        return_tensors="pt",
        truncation=True,
        max_length=int(os.getenv("SUMMARY_PT_CTX_MAX", "1024"))
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    procs = _legacy_build_logits_processors(
        tok_en,
        min_len=int(os.getenv("SUMMARY_GEN_MIN_NEW_TOKENS", "60")),
        rep_penalty=float(os.getenv("SUMMARY_GEN_REPETITION_PENALTY", "1.25")),
        ngram=int(os.getenv("SUMMARY_GEN_NO_REPEAT_NGRAM_SIZE", "4")),
        enc_input_ids=enc.get("input_ids")
    )
    out_ids = model_en.generate(
        **enc,
        max_new_tokens=int(os.getenv("SUMMARY_GEN_MAX_NEW_TOKENS", "224")),
        num_beams=int(os.getenv("SUMMARY_GEN_NUM_BEAMS", "4")),
        do_sample=os.getenv("SUMMARY_GEN_DO_SAMPLE", "false").lower() == "true",
        logits_processor=procs,
        early_stopping=True,
        pad_token_id=tok_en.pad_token_id,
        bos_token_id=getattr(tok_en, "bos_token_id", None),
        eos_token_id=getattr(tok_en, "eos_token_id", None),
    )
    summary_en = tok_en.decode(out_ids[0], skip_special_tokens=True).strip()

    # 9) EN -> FR protégé
    summary_fr = translate_en_to_fr_protected(summary_en, cfg, max_length=trans_max_len)

    # 10) Logs & retour
    dur = time.time() - t0
    log.info("feedback intent=%s topk=%d spans=%d len_en=%d time=%.2fs",
             intent, topk_k, len(spans_fr), len(summary_en.split()), dur)

    return {
        "ok": True,
        "intent": intent,
        "k": topk_k,
        "spans_fr": spans_fr,
        "joined_en": joined_en,
        "summary_en": summary_en,
        "summary_fr": summary_fr,
        "time_sec": round(dur, 2),
    }




