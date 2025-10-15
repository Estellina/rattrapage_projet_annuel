# utils.py
from __future__ import annotations
import re, math
from collections import Counter
from typing import List, Tuple
import numpy as np
import io, re
from typing import Tuple, List, Optional, Any
import torch

from langdetect import detect
from pypdf import PdfReader
import re
import unicodedata
from typing import Dict, Tuple, List, Set
import math, re
from collections import Counter
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer  # léger et efficace
import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)


import torch
from typing import Optional


_MARIAN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────
# Regex utilitaires existants (conservés)
_WS = re.compile(r"[\t\r\f]+")
_MULTI_NL = re.compile(r"\n{3,}")
_BAD = re.compile(r"[\u200b\u200c\u200d\ufeff]")


_MARIAN_MODEL = None
_MARIAN_TOKENIZER = None
_MARIAN_NAME = "Helsinki-NLP/opus-mt-fr-en"

import spacy
from spacy.language import Language

def make_spacy(lang: str) -> Language:
    # essaye le modèle natif si dispo
    try:
        if lang == "fr":
            return spacy.load("fr_core_news_sm", exclude=["ner","tagger","textcat","lemmatizer"])
        elif lang == "en":
            return spacy.load("en_core_web_sm", exclude=["ner","tagger","textcat","lemmatizer"])
    except OSError:
        pass

    # fallback léger et portable
    nlp = spacy.blank("fr" if lang == "fr" else "en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")  # règle de base: découpe aux ponctuations
    return nlp


def _get_marian(model_name: str):
    global _MARIAN_MODEL, _MARIAN_TOKENIZER, _MARIAN_NAME
    if _MARIAN_MODEL is None or _MARIAN_TOKENIZER is None or _MARIAN_NAME != model_name:
        _MARIAN_NAME = model_name
        from transformers import MarianMTModel, MarianTokenizer
        _MARIAN_TOKENIZER = MarianTokenizer.from_pretrained(model_name)
        _MARIAN_MODEL = MarianMTModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MARIAN_MODEL.to(device).eval()
    return _MARIAN_MODEL, _MARIAN_TOKENIZER


import os, re, json, tempfile, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

def detect_lang_simple(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "en"
    fr = 0; en = 0
    if any(c in t for c in "éèêàùûôîç"): fr += 2
    if re.search(r"\b(l'|d'|qu'|j'|n')", t): fr += 1
    if re.search(r"\b(can't|don't|it's|you're|we're)\b", t): en += 1
    FR_SW = {" le "," la "," les "," des "," du "," un "," une "," et "," ou "," de "," dans "," pour "," avec "," sur "," par "," au "," aux "," en "," est "," sont "}
    EN_SW = {" the "," and "," or "," of "," in "," to "," for "," with "," on "," by "," is "," are "," was "," were "}
    fr += sum(1 for w in FR_SW if w in f" {t} ")
    en += sum(1 for w in EN_SW if w in f" {t} ")
    return "fr" if fr > en else "en"

def _score_chunk_en(text: str) -> float:
    t = (text or "").lower().strip()
    if not t:
        return 0.0
    digits = sum(c.isdigit() for c in t)
    toks = t.split()
    uniq = len(set(toks)) / max(1, len(toks))
    return 0.65 * min(digits / 50.0, 1.0) + 0.35 * uniq

def _token_chunks_from_text(tok, text: str, max_t: int, overlap: int) -> List[List[int]]:
    if not text or not text.strip():
        return []
    enc = tok(text, add_special_tokens=False, return_attention_mask=False)
    ids = enc["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        # flatten in case of batched encoding
        ids = [t for row in ids for t in row]
    step = max(1, max_t - min(overlap, max_t // 3))
    out = []
    for s in range(0, len(ids), step):
        w = ids[s:s + max_t]
        if not w:
            break
        out.append(w)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Extraction PDF existante (conservée telle quelle)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t:
            chunks.append(t)
    return "\n\n".join(chunks)

# ➕ Nouvelle extraction robuste (pypdf -> pdfminer fallback) — optionnelle
def extract_text_safe(file_bytes: bytes) -> str:
    """
    Extraction robuste : tente pypdf puis bascule sur pdfminer.six en fallback.
    N'altère pas vos appels existants ; utilisez-la là où vous voulez plus de tolérance.
    """
    if not file_bytes:
        return ""
    # 1) pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                parts.append(txt)
        txt = "\n".join(parts).strip()
        if txt:
            return txt
    except Exception:
        pass
    # 2) pdfminer fallback
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return (pdfminer_extract(io.BytesIO(file_bytes)) or "").strip()
    except Exception:
        return ""

def chunked_generate_text(
    model,
    tokenizer,
    text: str,
    max_tokens: int,
    overlap: int,
    gen_kwargs: dict | None = None,
) -> str:
    """
    1) Tokenise le texte complet
    2) Découpe en fenêtres glissantes (max_tokens/overlap)
    3) Génère chunk par chunk → concatène
    Compatible encodeur-décodeur (mBART, Marian, T5).
    """
    if not text or not text.strip():
        return ""

    if gen_kwargs is None:
        gen_kwargs = dict(max_new_tokens=256, num_beams=2, early_stopping=True)

    # Device + eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.to(device).eval()
        # petit boost si GPU
        if torch.cuda.is_available():
            try:
                model.half()
            except Exception:
                pass
    except Exception:
        pass

    # Découpage tokenisé (nécessite chunk_text_with_context déjà présent dans utils)
    chunks_ids = chunk_text_with_context(
        text, tokenizer, max_tokens=max_tokens, overlap=overlap
    )
    if not chunks_ids:
        return ""

    out_texts: list[str] = []
    for ids in chunks_ids:
        # transformers >=4: accepte Tensor (B, L)
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            out_ids = model.generate(input_ids=input_ids, **gen_kwargs)
        txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if txt:
            out_texts.append(txt.strip())

    return " ".join(out_texts).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Nettoyage existant (conservé)
def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _BAD.sub("", text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()

# ──────────────────────────────────────────────────────────────────────────────
# ➕ Aides de découpage tokenisé avec overlap (agnostiques du tokenizer)

def ids_chunks_with_overlap(ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    """
    Découpe une liste d'ids en fenêtres glissantes avec chevauchement.
    """
    if not ids:
        return []
    if len(ids) <= max_tokens:
        return [ids]
    if overlap >= max_tokens:
        # sécurités soft : on impose un overlap < max_tokens
        overlap = max(0, max_tokens - 1)
    chunks: List[List[int]] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunks.append(ids[start:end])
        if end >= len(ids):
            break
        start = max(0, end - overlap)
    return chunks

def chunk_text_with_context(
    text: str,
    tokenizer: Any,
    max_tokens: int = 1024,
    overlap: int = 200
) -> List[List[int]]:
    """
    Essaie d'être compatible avec :
    - tokenizers HF (tokenizer(...)/encode, avec ou sans return_tensors)
    - sentencepiece-like (encode().ids)
    Retourne une liste de listes d'ids.
    """
    if not text or text.strip() == "":
        return []
    # Obtenir des ids de manière robuste
    ids: Optional[List[int]] = None
    try:
        enc = tokenizer(text, return_tensors=None, add_special_tokens=True, truncation=False)
        if isinstance(enc, dict) and "input_ids" in enc:
            ids = enc["input_ids"]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
        elif isinstance(enc, list):
            ids = enc
    except Exception:
        pass

    if ids is None:
        # Fallback : interface encode() / encode().ids
        if hasattr(tokenizer, "encode"):
            try:
                obj = tokenizer.encode(text)
                ids = getattr(obj, "ids", obj)  # sentencepiece: has .ids ; HF encode -> list
            except Exception:
                # dernier essai HF
                ids = tokenizer.encode(text, add_special_tokens=True)
        else:
            raise RuntimeError("Tokenizer incompatible : impossible d'encoder le texte.")

    if not isinstance(ids, list):
        # si Tensor ou autre structure
        try:
            ids = ids[0].tolist()
        except Exception:
            ids = list(ids)

    return ids_chunks_with_overlap(ids, max_tokens=max_tokens, overlap=overlap)

# ──────────────────────────────────────────────────────────────────────────────
# ➕ Génération chunkée générique (n'importe quel modèle encoder-decoder HF)



def _translate_chunked_marian(
    text: str,
    model_id: str,
    # fenêtres un peu plus courtes → encodage + génération plus rapides
    max_tokens: int = 440,     # (ex-480) < 512 pour Marian
    overlap: int = 48,         # (ex-64)
    gen_kwargs: Optional[dict] = None,
) -> str:
    """
    Charge Marian (tokenizer+model) une fois (cache) et traduit en mode chunké.
    - GPU si dispo (FP16 quand possible)
    - early_stopping=True par défaut
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not text or not text.strip():
        return text

    tok = _translators.get((model_id, "tok"))
    mdl = _translators.get((model_id, "mdl"))
    if tok is None or mdl is None:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        # move to device + eval + FP16 si GPU
        mdl.to(_MARIAN_DEVICE).eval()
        if torch.cuda.is_available():
            try:
                mdl.half()
            except Exception:
                pass
        _translators[(model_id, "tok")] = tok
        _translators[(model_id, "mdl")] = mdl

    if gen_kwargs is None:
        gen_kwargs = dict(max_new_tokens=256, early_stopping=True, num_beams=2)

    return chunked_generate_text(
        _translators[(model_id, "mdl")],
        _translators[(model_id, "tok")],
        text,
        max_tokens=max_tokens,
        overlap=overlap,
        gen_kwargs=gen_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Traduction (existante) — nous la conservons, mais nous allons la fiabiliser avec un chemin chunké

_translators = {}

# ➕ Traduction Marian chunkée (sécurisée) utilisée par maybe_translate si possible

def maybe_translate(text: str, target_lang: str = "en", enable_offline: bool = False) -> Tuple[str, str]:
    """
    ⚠️ Signature/contrat conservés : (translated_text, detected_lang)
    - Détection langue via langdetect (comme avant).
    - Si enable_offline=False OU si déjà dans la langue cible → retourne tel quel.
    - Sinon, utilise Marian **chunké** (plus robuste que la version monolithique).
    """
    try:
        lang = detect(text) if text.strip() else target_lang
    except Exception:
        lang = target_lang

    # Si pas besoin de traduire ou traduction offline désactivée → comportement inchangé
    if not enable_offline or lang.lower().startswith(target_lang.lower()):
        return text, lang

    # Choix du modèle Marian (mêmes mappings que votre version, mais passage chunké)
    if lang.startswith("fr") and target_lang.startswith("en"):
        model_id = "Helsinki-NLP/opus-mt-fr-en"
    elif lang.startswith("en") and target_lang.startswith("fr"):
        model_id = "Helsinki-NLP/opus-mt-en-fr"
    else:
        model_id = "Helsinki-NLP/opus-mt-mul-en" if target_lang.startswith("en") else "Helsinki-NLP/opus-mt-en-mul"

    # Traduction chunkée sécurisée (évite IndexError: index out of range in self)
    try:
        translated = _translate_chunked_marian(
            text,
            model_id=model_id,
            max_tokens=480,
            overlap=64,
            gen_kwargs=dict(max_new_tokens=256),
        )
        return translated, lang
    except Exception:
        # 🔁 Fallback : votre ancienne logique monolithique (préservée)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = _translators.get((lang, target_lang, "tok"))
        mdl = _translators.get((lang, target_lang, "mdl"))
        if tok is None or mdl is None:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            _translators[(lang, target_lang, "tok")] = tok
            _translators[(lang, target_lang, "mdl")] = mdl
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
        out = mdl.generate(**inputs, max_new_tokens=1024)
        trans = tok.batch_decode(out, skip_special_tokens=True)[0]
        return trans, lang

# ──────────────────────────────────────────────────────────────────────────────
# Score qualité existant (conservé)
# Stopwords FR/EN minimalistes (pour rester sans dépendances)
_STOP_FR = {
    "le","la","les","un","une","des","de","du","au","aux","et","ou","mais","donc","or","ni","car",
    "dans","en","sur","sous","par","pour","avec","sans","entre","chez","vers","plus","moins","comme",
    "que","qui","quoi","dont","où","ce","cet","cette","ces","ça","cela","c","d","l","n","qu","s","t",
    "est","sont","ai","as","avons","avez","ont","étais","était","étions","étiez","étaient","être",
    "été","avoir","fait","faites","fais","faisons","faites","font","y","aujourd","hui"
}
_STOP_EN = {
    "the","a","an","and","or","but","so","nor","for","of","to","in","on","at","by","with","from",
    "as","that","which","who","whom","this","these","those","it","its","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","not","no","yes","if","then","than","into",
    "over","under","between","within","without","about","through","across","per","via"
}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-']{1,}")

def _normalize_pdf_text(txt: str) -> str:
    """Normalisation douce pour texte PDF: espaces, ligatures, césures, unicode."""
    if not txt:
        return ""
    # NFKC + suppression zéros-width
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"[\u200B-\u200D\uFEFF]", "", txt)

    # Césures de fin de ligne : "infor-\nmation" -> "information"
    txt = re.sub(r"-\s*\n\s*", "", txt)

    # Remplacer multiples espaces/saute-lignes par formes propres
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t\f]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

def _remove_stop(words: List[str]) -> List[str]:
    # Heuristique bilingue: retire stop FR/EN
    return [w for w in words if w not in _STOP_FR and w not in _STOP_EN]

def _precision_recall_f1(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, float, float]:
    """ROUGE-1-like simple (sur tokens filtrés)."""
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1

def _novelty_penalty(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """Pénalise le copier-coller pur (faible nouveauté) mais sans punir un bon rappel."""
    if not hyp_tokens:
        return 0.3
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    novelty = len([w for w in hyp_set if w not in ref_set]) / max(1, len(hyp_set))
    # Plus la nouveauté est basse (<10%), plus la pénalité monte jusqu'à 0.25
    if novelty >= 0.25:
        return 0.0
    if novelty >= 0.10:
        return 0.1
    return 0.25

def _ngram_repetition_rate(tokens: List[str], n: int = 3) -> float:
    """Taux de répétition de n-grammes (0 = pas de répétition)."""
    if len(tokens) < n * 2:
        return 0.0
    counts = {}
    total = 0
    for i in range(len(tokens) - n + 1):
        total += 1
        key = tuple(tokens[i:i+n])
        counts[key] = counts.get(key, 0) + 1
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / max(1, total)

def _length_ratio_penalty(src_len: int, hyp_len: int, low: float, high: float) -> float:
    """Pénalise si |hyp| est < low*|src| ou > high*|src| (fenêtre attendue)."""
    if src_len == 0 or hyp_len == 0:
        return 0.3
    ratio = hyp_len / src_len
    if low <= ratio <= high:
        return 0.0
    # en dehors, pénalité progressive jusqu'à 0.3
    dist = 0.0
    if ratio < low:
        dist = (low - ratio) / low
    else:
        dist = (ratio - high) / high
    return max(0.0, min(0.3, dist * 0.3))

def _structure_bonus(text: str) -> float:
    """Bonus léger si la sortie a une structure (phrases, puces, paragraphes)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = sum(1 for l in lines if re.match(r"^[-–—•\*\d]+\s", l))
    sentences = re.split(r"[.!?]\s+", text.strip())
    para = text.count("\n\n")
    bonus = 0.0
    if bullets >= 2:
        bonus += 0.05
    if para >= 1:
        bonus += 0.05
    if len(sentences) >= 3:
        bonus += 0.05
    return min(0.12, bonus)

def simple_quality_score(source: str, generated: str) -> float:
    """
    Score heuristique ∈ [0,1] optimisé pour des 'sources' issues de PDF :
      - Normalisation PDF (césures/espaces/ligatures)
      - ROUGE-1 simplifié (précision/recall/F1 sur tokens filtrés)
      - Pénalité copie brute (faible nouveauté)
      - Pénalité répétition n-gram
      - Pénalité longueur relative (par défaut 5%–30% de la source)
      - Bonus structure (phrases, puces, paragraphes)
    Pondérations choisies empiriquement pour un bon compromis lisibilité/fidélité.
    """
    gen = _normalize_pdf_text(generated)
    if not gen:
        return 0.0
    src = _normalize_pdf_text(source)

    # Tokenisation + filtrage stopwords
    src_tokens_all = _tokenize(src)
    gen_tokens_all = _tokenize(gen)
    src_tokens = _remove_stop(src_tokens_all)
    gen_tokens = _remove_stop(gen_tokens_all)

    # Couverture façon ROUGE-1
    p, r, f1 = _precision_recall_f1(src_tokens, gen_tokens)

    # Pénalités / bonus
    pen_novel = _novelty_penalty(src_tokens, gen_tokens)          # 0 → 0.25
    rep_rate = _ngram_repetition_rate(gen_tokens, n=3)            # 0 → 1
    pen_repeat = min(0.3, rep_rate * 0.6)                         # jusqu’à 0.3
    pen_len = _length_ratio_penalty(len(src_tokens_all), len(gen_tokens_all),
                                    low=0.05, high=0.30)          # résumé 5%–30% par défaut
    bonus_struct = _structure_bonus(gen)                           # 0 → 0.12

    # Agrégation (pondérations)
    # Base: 0.55*F1 + 0.15*P + 0.10*R
    base = 0.55*f1 + 0.15*p + 0.10*r

    # Retrait des pénalités + ajout bonus
    score = base - pen_novel - pen_repeat - pen_len + bonus_struct

    # Clamp [0,1]
    return max(0.0, min(1.0, score))

def simple_quality_breakdown(source: str, generated: str) -> Dict[str, float]:
    """
    Version détaillée pour debug/monitoring.
    """
    gen = _normalize_pdf_text(generated)
    src = _normalize_pdf_text(source)

    src_tokens_all = _tokenize(src)
    gen_tokens_all = _tokenize(gen)
    src_tokens = _remove_stop(src_tokens_all)
    gen_tokens = _remove_stop(gen_tokens_all)

    p, r, f1 = _precision_recall_f1(src_tokens, gen_tokens)
    pen_novel = _novelty_penalty(src_tokens, gen_tokens)
    rep_rate = _ngram_repetition_rate(gen_tokens, n=3)
    pen_repeat = min(0.3, rep_rate * 0.6)
    pen_len = _length_ratio_penalty(len(src_tokens_all), len(gen_tokens_all), low=0.05, high=0.30)
    bonus_struct = _structure_bonus(gen)

    base = 0.55*f1 + 0.15*p + 0.10*r
    score = max(0.0, min(1.0, base - pen_novel - pen_repeat - pen_len + bonus_struct))

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "rep_rate_3gram": round(rep_rate, 4),
        "pen_novelty": round(pen_novel, 4),
        "pen_repeat": round(pen_repeat, 4),
        "pen_length": round(pen_len, 4),
        "bonus_structure": round(bonus_struct, 4),
        "final_score": round(score, 4),
        "src_tokens": len(src_tokens_all),
        "gen_tokens": len(gen_tokens_all),
    }

# --- Scoring utils ------------------------------------------------------------

# app/pipelines/summary_utils.py


# ---------- tokenisation & métriques de surface ----------
_WORD_RE = re.compile(r"\w+", re.UNICODE)

def tokens(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

def distinct_n(s: str, n: int = 2) -> float:
    toks = tokens(s)
    if len(toks) < n:
        return 1.0
    ngrams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(ngrams)) / max(1, len(ngrams))

def sentence_well_formed_ratio(s: str) -> float:
    sents = re.split(r"(?<=[\.!?])\s+", (s or "").strip())
    if not sents:
        return 0.0
    good = 0
    for x in sents:
        x = x.strip()
        if not x:
            continue
        cap = x[:1].isupper()
        end = bool(re.search(r"[\.!?]$", x))
        good += 1 if (cap and end and len(x) > 20) else 0
    return good / max(1, len(sents))

def punctuation_density(s: str) -> float:
    p = sum(1 for c in (s or "") if c in ",;:.!?")
    w = max(1, len(tokens(s)))
    return p / w  # ~0.03–0.08 raisonnable

def repetition_penalty_proxy(s: str) -> float:
    # 1.0 = pas de répétition ; <1.0 = répétitif
    return 0.5 * distinct_n(s, 2) + 0.5 * distinct_n(s, 3)

def translation_quality_heur(fr_src: str, en_hyp: str) -> float:
    """Heuristique simple FR->EN combinant ratios, ponctuation, phrases complètes et non-répétition."""
    len_fr = max(1, len(tokens(fr_src)))
    len_en = max(1, len(tokens(en_hyp)))
    ratio = len_en / len_fr
    ratio_score = 1.0 - min(abs(math.log(ratio)), 1.0)  # 1 si ratio≈1
    punc = punctuation_density(en_hyp)
    punc_score = max(0.0, min(1.0, (punc - 0.02) / 0.06))
    sent_score = sentence_well_formed_ratio(en_hyp)
    rep_score = repetition_penalty_proxy(en_hyp)
    return 0.35*ratio_score + 0.2*punc_score + 0.25*sent_score + 0.2*rep_score

# ---------- similarité / pertinence ----------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))

def _relevance_scores_tfidf(chunks_en: List[str], context_en: str) -> List[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # optionnel
        vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
        X = vect.fit_transform([context_en] + chunks_en).astype(np.float32)
        ctx = X[0].toarray()[0]
        sims = []
        for i in range(1, X.shape[0]):
            sims.append(cosine_sim(ctx, X[i].toarray()[0]))
        return sims
    except Exception:
        # Fallback BoW Jaccard (zéro dépendance)
        ctx = set(tokens(context_en))
        sims = []
        for ch in chunks_en:
            t = set(tokens(ch))
            inter = len(ctx & t); union = len(ctx | t)
            sims.append(inter / union if union else 0.0)
        return sims

def score_chunks(
    fr_chunks: List[str],
    en_chunks: List[str],
    context_en: str,
    w_sim: float = 0.6,
    w_qlt: float = 0.4,
) -> List[Tuple[int, float]]:
    """Retourne [(index_chunk, score)] trié décroissant."""
    sims = _relevance_scores_tfidf(en_chunks, context_en)
    scored = []
    for i, en in enumerate(en_chunks):
        q = translation_quality_heur(fr_chunks[i], en)
        s = w_sim * sims[i] + w_qlt * q
        scored.append((i, float(s)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ---------- dé-duplication / nettoyage simple ----------
def dedup_texts_keep_order(texts: List[str]) -> List[str]:
    seen, out = set(), []
    for t in texts:
        tt = (t or "").strip()
        if not tt or tt in seen:
            continue
        seen.add(tt); out.append(tt)
    return out

