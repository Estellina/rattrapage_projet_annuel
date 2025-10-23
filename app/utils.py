from __future__ import annotations

"""
utils.py — version nettoyée et organisée
- Imports dédoublonnés et triés
- Constantes et regex centralisées
- Fonctions regroupées par domaine (PDF, tokenisation/decoupage, traduction, nettoyage, scoring)
- Types et docstrings ajoutés/lissés
- Compatibilité conservée avec l'existant
"""

# ============================================================================
# Imports
# ============================================================================
import io
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from langdetect import detect
from pypdf import PdfReader

# Modules optionnels (chargés à l'usage)
# - spacy pour la segmentation (fallback en blank avec sentencizer)
# - sklearn pour TF-IDF (fallback Jaccard si non dispo)
import spacy
from spacy.language import Language

# ============================================================================
# Constantes / Regex / Stopwords
# ============================================================================

# Device par défaut pour les modèles (CPU/GPU)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nom du modèle Marian par défaut
_MARIAN_NAME = "Helsinki-NLP/opus-mt-fr-en"

# Cache simple pour traducteurs/tokenizers
_translators: Dict[Tuple[str, str], Any] = {}
_translators_tok: Dict[str, Any] = {}
_translators_mdl: Dict[str, Any] = {}

# Espaces/saute-lignes indésirables et caractères zéro-width
_RE_WS = re.compile(r"[\t\r\f]+")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_BAD = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# Token regex bilingue FR/EN
_RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-']{1,}")

# Stopwords minimalistes FR/EN
_STOP_FR: Set[str] = {
    "le","la","les","un","une","des","de","du","au","aux","et","ou","mais","donc","or","ni","car",
    "dans","en","sur","sous","par","pour","avec","sans","entre","chez","vers","plus","moins","comme",
    "que","qui","quoi","dont","où","ce","cet","cette","ces","ça","cela","c","d","l","n","qu","s","t",
    "est","sont","ai","as","avons","avez","ont","étais","était","étions","étiez","étaient","être",
    "été","avoir","fait","faites","fais","faisons","font","y","aujourd","hui",
}
_STOP_EN: Set[str] = {
    "the","a","an","and","or","but","so","nor","for","of","to","in","on","at","by","with","from",
    "as","that","which","who","whom","this","these","those","it","its","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","not","no","yes","if","then","than","into",
    "over","under","between","within","without","about","through","across","per","via",
}

# ============================================================================
# spaCy utilitaire
# ============================================================================

def make_spacy(lang: str) -> Language:
    """Retourne un pipeline spaCy léger pour *lang* ("fr"|"en").
    Essaye d'abord un modèle natif, sinon fallback sur blank+sentencizer.
    """
    try:
        if lang == "fr":
            return spacy.load("fr_core_news_sm", exclude=["ner", "tagger", "textcat", "lemmatizer"])
        if lang == "en":
            return spacy.load("en_core_web_sm", exclude=["ner", "tagger", "textcat", "lemmatizer"])
    except OSError:
        pass

    nlp = spacy.blank("fr" if lang == "fr" else "en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

# ============================================================================
# Détection de langue (heuristique très simple, offline)
# ============================================================================

def detect_lang_simple(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "en"
    fr = 0
    en = 0
    if any(c in t for c in "éèêàùûôîç"):  # accents FR
        fr += 2
    if re.search(r"\b(l'|d'|qu'|j'|n')", t):
        fr += 1
    if re.search(r"\b(can't|don't|it's|you're|we're)\b", t):
        en += 1

    fr_sw = {" le "," la "," les "," des "," du "," un "," une "," et "," ou "," de "," dans "," pour "," avec "," sur "," par "," au "," aux "," en "," est "," sont "}
    en_sw = {" the "," and "," or "," of "," in "," to "," for "," with "," on "," by "," is "," are "," was "," were "}
    fr += sum(1 for w in fr_sw if w in f" {t} ")
    en += sum(1 for w in en_sw if w in f" {t} ")
    return "fr" if fr > en else "en"

# ============================================================================
# PDF — extraction & normalisation
# ============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extraction simple du texte via pypdf (comme avant)."""
    if not file_bytes:
        return ""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks: List[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t:
            chunks.append(t)
    return "\n\n".join(chunks)


def extract_text_safe(file_bytes: bytes) -> str:
    """Extraction robuste : pypdf → fallback pdfminer.six.
    N'altère pas vos appels existants ; utilisez-la quand vous voulez plus de tolérance.
    """
    if not file_bytes:
        return ""
    # 1) pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts: List[str] = []
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


def _normalize_pdf_text(txt: str) -> str:
    """Normalisation douce pour texte PDF: espaces, ligatures, césures, unicode."""
    if not txt:
        return ""
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"[\u200B-\u200D\uFEFF]", "", txt)  # zéro-width
    txt = re.sub(r"-\s*\n\s*", "", txt)  # césures fin de ligne
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t\f]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# ============================================================================
# Nettoyage texte
# ============================================================================

def clean_text(text: str) -> str:
    """Nettoyage standard : espaces, ponctuation collée, multi sauts de ligne."""
    text = (text or "").replace("\xa0", " ")
    text = _RE_BAD.sub("", text)
    text = _RE_WS.sub(" ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\\1", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    return text.strip()

# ============================================================================
# Tokenisation / découpage avec chevauchement
# ============================================================================

def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _RE_WORD.finditer(text or "")]


def ids_chunks_with_overlap(ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    """Découpe une liste d'ids en fenêtres glissantes avec chevauchement."""
    if not ids:
        return []
    if len(ids) <= max_tokens:
        return [ids]
    if overlap >= max_tokens:
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
    overlap: int = 200,
) -> List[List[int]]:
    """Retourne des *ids* tokenisés avec découpage chevauchant. Compatible HF/spm."""
    if not text or text.strip() == "":
        return []

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
        if hasattr(tokenizer, "encode"):
            try:
                obj = tokenizer.encode(text)
                ids = getattr(obj, "ids", obj)
            except Exception:
                ids = tokenizer.encode(text, add_special_tokens=True)
        else:
            raise RuntimeError("Tokenizer incompatible : impossible d'encoder le texte.")

    if not isinstance(ids, list):
        try:
            ids = ids[0].tolist()
        except Exception:
            ids = list(ids)

    return ids_chunks_with_overlap(ids, max_tokens=max_tokens, overlap=overlap)

# ============================================================================
# Génération/Traduction chunkée (transformers)
# ============================================================================

def chunked_generate_text(
    model: Any,
    tokenizer: Any,
    text: str,
    max_tokens: int,
    overlap: int,
    gen_kwargs: Optional[dict] = None,
) -> str:
    """Génère du texte chunk par chunk à partir d'un modèle seq2seq HF."""
    if not text or not text.strip():
        return ""

    if gen_kwargs is None:
        gen_kwargs = dict(max_new_tokens=256, num_beams=2, early_stopping=True)

    try:
        model.to(_DEVICE).eval()
        if torch.cuda.is_available():
            try:
                model.half()
            except Exception:
                pass
    except Exception:
        pass

    chunks_ids = chunk_text_with_context(text, tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not chunks_ids:
        return ""

    out_texts: List[str] = []
    for ids in chunks_ids:
        input_ids = torch.tensor([ids], device=_DEVICE)
        with torch.no_grad():
            out_ids = model.generate(input_ids=input_ids, **gen_kwargs)
        txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if txt:
            out_texts.append(txt.strip())

    return " ".join(out_texts).strip()


def _translate_chunked_marian(
    text: str,
    model_id: str,
    max_tokens: int = 480,
    overlap: int = 64,
    gen_kwargs: Optional[dict] = None,
) -> str:
    """Charge Marian (tokenizer+model) au besoin et traduit en mode chunké."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not text or not text.strip():
        return text

    tok = _translators_tok.get(model_id)
    mdl = _translators_mdl.get(model_id)
    if tok is None or mdl is None:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        mdl.to(_DEVICE).eval()
        if torch.cuda.is_available():
            try:
                mdl.half()
            except Exception:
                pass
        _translators_tok[model_id] = tok
        _translators_mdl[model_id] = mdl

    if gen_kwargs is None:
        gen_kwargs = dict(max_new_tokens=256, early_stopping=True, num_beams=2)

    return chunked_generate_text(mdl, tok, text, max_tokens=max_tokens, overlap=overlap, gen_kwargs=gen_kwargs)


def maybe_translate(text: str, target_lang: str = "en", enable_offline: bool = False) -> Tuple[str, str]:
    """Retourne (texte_traduit, langue_detectée).
    - Détection via langdetect (fallback target_lang en cas d'erreur).
    - Si enable_offline=False ou si *text* est déjà dans la langue cible → retourne tel quel.
    - Sinon, utilise Marian en découpage chunké pour robustesse.
    """
    try:
        lang = detect(text) if text.strip() else target_lang
    except Exception:
        lang = target_lang

    if not enable_offline or lang.lower().startswith(target_lang.lower()):
        return text, lang

    if lang.startswith("fr") and target_lang.startswith("en"):
        model_id = "Helsinki-NLP/opus-mt-fr-en"
    elif lang.startswith("en") and target_lang.startswith("fr"):
        model_id = "Helsinki-NLP/opus-mt-en-fr"
    else:
        model_id = "Helsinki-NLP/opus-mt-mul-en" if target_lang.startswith("en") else "Helsinki-NLP/opus-mt-en-mul"

    try:
        translated = _translate_chunked_marian(text, model_id=model_id, max_tokens=480, overlap=64, gen_kwargs=dict(max_new_tokens=256))
        return translated, lang
    except Exception:
        # Fallback monolithique si generate() chunké échoue
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok_key = f"{model_id}::tok"
        mdl_key = f"{model_id}::mdl"
        tok = _translators_tok.get(tok_key)
        mdl = _translators_mdl.get(mdl_key)
        if tok is None or mdl is None:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            _translators_tok[tok_key] = tok
            _translators_mdl[mdl_key] = mdl
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
        out = mdl.generate(**inputs, max_new_tokens=1024)
        trans = tok.batch_decode(out, skip_special_tokens=True)[0]
        return trans, lang

# ============================================================================
# Scores qualité & métriques de surface
# ============================================================================

def _remove_stop(words: List[str]) -> List[str]:
    return [w for w in words if w not in _STOP_FR and w not in _STOP_EN]


def _precision_recall_f1(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, float, float]:
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


def _novelty_penalty(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not hyp_tokens:
        return 0.3
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    novelty = len([w for w in hyp_set if w not in ref_set]) / max(1, len(hyp_set))
    if novelty >= 0.25:
        return 0.0
    if novelty >= 0.10:
        return 0.1
    return 0.25


def _ngram_repetition_rate(tokens: List[str], n: int = 3) -> float:
    if len(tokens) < n * 2:
        return 0.0
    counts: Dict[Tuple[str, ...], int] = {}
    total = 0
    for i in range(len(tokens) - n + 1):
        total += 1
        key = tuple(tokens[i:i + n])
        counts[key] = counts.get(key, 0) + 1
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / max(1, total)


def _length_ratio_penalty(src_len: int, hyp_len: int, low: float, high: float) -> float:
    if src_len == 0 or hyp_len == 0:
        return 0.3
    ratio = hyp_len / src_len
    if low <= ratio <= high:
        return 0.0
    dist = (low - ratio) / low if ratio < low else (ratio - high) / high
    return max(0.0, min(0.3, dist * 0.3))


def _structure_bonus(text: str) -> float:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    bullets = sum(1 for l in lines if re.match(r"^[-–—•\*\d]+\s", l))
    sentences = re.split(r"[.!?]\s+", (text or "").strip())
    para = (text or "").count("\n\n")
    bonus = 0.0
    if bullets >= 2:
        bonus += 0.05
    if para >= 1:
        bonus += 0.05
    if len(sentences) >= 3:
        bonus += 0.05
    return min(0.12, bonus)


def simple_quality_score(source: str, generated: str) -> float:
    gen = _normalize_pdf_text(generated)
    if not gen:
        return 0.0
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

    base = 0.55 * f1 + 0.15 * p + 0.10 * r
    score = base - pen_novel - pen_repeat - pen_len + bonus_struct
    return max(0.0, min(1.0, score))


def simple_quality_breakdown(source: str, generated: str) -> Dict[str, float]:
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

    base = 0.55 * f1 + 0.15 * p + 0.10 * r
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

# ============================================================================
# Métriques supplémentaires & pertinence
# ============================================================================

def tokens(s: str) -> List[str]:
    return re.findall(r"\w+", (s or "").lower(), flags=re.UNICODE)


def distinct_n(s: str, n: int = 2) -> float:
    toks = tokens(s)
    if len(toks) < n:
        return 1.0
    ngrams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
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
    return p / w


def repetition_penalty_proxy(s: str) -> float:
    return 0.5 * distinct_n(s, 2) + 0.5 * distinct_n(s, 3)


def translation_quality_heur(fr_src: str, en_hyp: str) -> float:
    len_fr = max(1, len(tokens(fr_src)))
    len_en = max(1, len(tokens(en_hyp)))
    ratio = len_en / len_fr
    ratio_score = 1.0 - min(abs(math.log(ratio)), 1.0)
    punc = punctuation_density(en_hyp)
    punc_score = max(0.0, min(1.0, (punc - 0.02) / 0.06))
    sent_score = sentence_well_formed_ratio(en_hyp)
    rep_score = repetition_penalty_proxy(en_hyp)
    return 0.35 * ratio_score + 0.2 * punc_score + 0.25 * sent_score + 0.2 * rep_score


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def _relevance_scores_tfidf(chunks_en: List[str], context_en: str) -> List[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
        X = vect.fit_transform([context_en] + chunks_en).astype(np.float32)
        ctx = X[0].toarray()[0]
        sims: List[float] = []
        for i in range(1, X.shape[0]):
            sims.append(cosine_sim(ctx, X[i].toarray()[0]))
        return sims
    except Exception:
        ctx = set(tokens(context_en))
        sims: List[float] = []
        for ch in chunks_en:
            t = set(tokens(ch))
            inter = len(ctx & t)
            union = len(ctx | t)
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
    scored: List[Tuple[int, float]] = []
    for i, en in enumerate(en_chunks):
        q = translation_quality_heur(fr_chunks[i], en)
        s = w_sim * sims[i] + w_qlt * q
        scored.append((i, float(s)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ============================================================================
# Dé-duplication simple
# ============================================================================

def dedup_texts_keep_order(texts: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in texts:
        tt = (t or "").strip()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        out.append(tt)
    return out
