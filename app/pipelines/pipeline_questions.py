# app/pipelines/pipeline_questions.py
# -----------------------------------------------------------------------------
# Pipeline Questions unifiée :
#  - charge UNE fois le modèle FLAN-T5 (S3 -> cache local -> HF fallback)
#  - génération standard: generate_questions_from_document(...)
#  - feedback/régénération: regenerate_questions_with_feedback(...)
#  - helpers d’extraction PDF + traduction via app.utils
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import re
import numpy as np
import time
import json
import logging
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

from app import utils
from app.config import cfg
from app.models.pretrained_qg_flan import (
    FlanT5QuestionGenerator,
    select_top_spans,
    postprocess_questions,
    keyword_tokens,
)

from app.pipelines.pipeline_summary import translate_en_to_fr, translate_fr_to_en, detect_lang_simple, clean_text_fr, clean_text_en

try:
    from app.pipelines.pipeline_summary import SumConfig as _SumConfig
except Exception:
    _SumConfig = None

log = logging.getLogger("pipeline_questions")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)


import re
from typing import Iterable, List

from dataclasses import dataclass, asdict
from typing import Literal

Reason = Literal[
    "off-topic","too-trivial","too-hard","unclear","unanswerable",
    "wrong-section","wrong-style","duplicate","too-long","too-short","needs-diversity"
]

@dataclass
class QGConstraintsState:
    passages_en: List[str]
    num_questions: int = 5
    difficulty: str = "intermediate"
    style: str = "exam"
    diversity: float = 0.45
    length: str = "medium"
    force_in_context: bool = True

_QG_SESS: Dict[str, QGConstraintsState] = {}

def _mk_doc_key(text_en: str) -> str:
    import hashlib
    return hashlib.sha1(text_en.encode("utf-8")).hexdigest()


def apply_feedback_rules(con: QGConstraintsState, reason: Reason, fields: Optional[Dict[str, Any]] = None) -> QGConstraintsState:
    c = QGConstraintsState(**asdict(con))
    fields = fields or {}

    if reason == "off-topic":
        c.force_in_context = True
        if "passages_en" in fields:
            c.passages_en = fields["passages_en"]

    elif reason == "too-trivial":
        c.difficulty = "advanced" if c.difficulty != "advanced" else c.difficulty
        c.diversity = min(1.0, c.diversity + 0.2)
        c.length = "medium"

    elif reason == "too-hard":
        c.difficulty = "easy"; c.length = "short"
        c.diversity = max(0.0, c.diversity - 0.2)

    elif reason == "unclear":
        c.style = "concise"; c.length = "short"

    elif reason == "unanswerable":
        c.force_in_context = True
        if "passages_en" in fields:
            c.passages_en = fields["passages_en"]

    elif reason == "wrong-section":
        if "passages_en" in fields:
            c.passages_en = fields["passages_en"]

    elif reason == "wrong-style":
        c.style = fields.get("style", "academic")

    elif reason == "duplicate":
        c.diversity = min(1.0, c.diversity + 0.3)

    elif reason == "too-long":
        c.length = "short"

    elif reason == "too-short":
        c.length = "long" if c.length != "long" else "medium"

    elif reason == "needs-diversity":
        c.diversity = min(1.0, c.diversity + 0.3)

    # override explicite si fourni
    for k in ["difficulty","style","diversity","length","num_questions","passages_en","force_in_context"]:
        if k in fields:
            setattr(c, k, fields[k])
    return c





# ───────────────────────────────────── ENV helpers
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

def _env_float(name: str, default: float) -> float:
    try:
        return float(getattr(cfg, name, os.getenv(name, default)))
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

# ───────────────────────────────────── S3 sync (identique à la version feedback)
def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Bad S3 URI: {uri!r}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix

# ---------------------------------------------------------------------
# Helpers pour utiliser la traduction de la pipeline Résumé
# ---------------------------------------------------------------------

def _get_sum_cfg():
    """
    Retourne un objet 'cfg' acceptable pour translate_fr_to_en/translate_en_to_fr.
    - Si la pipeline Résumé expose SumConfig sans arguments obligatoires, on instancie.
    - Sinon, on fournit un petit stub avec au moins 'trans_max_length'.
    """
    # 1) Essaye SumConfig() si dispo et sans args
    if _SumConfig is not None:
        try:
            return _SumConfig()  # la plupart du temps ça marche tel quel
        except TypeError:
            pass

    # 2) Stub minimal (les fonctions n'ont souvent besoin que de trans_max_length)
    class _Stub:
        pass
    stub = _Stub()
    try:
        setattr(stub, "trans_max_length", int(os.getenv("TRANS_MAX_LENGTH", "480")))
    except Exception:
        setattr(stub, "trans_max_length", 480)
    return stub

_FR_SIGNS = (" à ", " é", " è", " ê", " ç", " de ", " du ", " des ", " la ", " le ", " les ",
             " un ", " une ", " au ", " aux ", " et ", " pour ", " avec ", " sur ", " dans ")

def _is_probably_french(t: str) -> bool:
    """
    Heuristique simple : présence d’accents + stopwords FR.
    Suffisant pour éviter une traduction FR->EN quand le texte est déjà en EN.
    """
    if not t:
        return False
    s = " " + t.lower() + " "
    score = 0
    for w in _FR_SIGNS:
        if w in s:
            score += 1
            if score >= 3:
                return True
    # accents
    return any(c in s for c in ("é", "è", "ê", "à", "ù", "ç"))


def _sync_s3_dir_to_local(s3_uri: str, dest_dir: str) -> bool:
    os.makedirs(dest_dir, exist_ok=True)
    # 1) boto3 si dispo
    try:
        import boto3  # type: ignore
        bucket, prefix = _parse_s3_uri(s3_uri)
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)
        paginator = s3.get_paginator("list_objects_v2")
        any_dl = False
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):].lstrip("/")
                out_path = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                # Skip si la même taille est déjà présente
                want = int(obj.get("Size", 0) or 0)
                if os.path.exists(out_path) and want > 0:
                    try:
                        have = os.path.getsize(out_path)
                        if have == want:
                            continue
                    except Exception:
                        pass

                s3.download_file(bucket, key, out_path)
                any_dl = True

        # Considère la sync réussie si on a téléchargé ou si le cache est déjà complet
        return any_dl or _has_required_files(dest_dir)
    except Exception as e:
        log.debug(f"[QG][s3] boto3 sync failed: {e}")

    # 2) aws cli
    try:
        subprocess.run(["aws", "s3", "sync", s3_uri, dest_dir, "--no-progress"], check=True)
        return _has_required_files(dest_dir)
    except Exception as e:
        log.debug(f"[QG][s3] aws cli sync failed: {e}")
        return _has_required_files(dest_dir)


def _has_required_files(cache_dir: str) -> bool:
    def _exists(*names) -> bool:
        return any(os.path.exists(os.path.join(cache_dir, n)) for n in names)
    return _exists("config.json") and _exists("tokenizer.json", "spiece.model") and _exists("pytorch_model.bin", "model.safetensors")

def _ensure_local_model_dir() -> str:
    s3_prefix = _env_str("PT_QG_S3_PREFIX", "")
    if not s3_prefix:
        return ""

    cache_dir = _env_str("PT_QG_CACHE_DIR") or os.path.join(
        os.getenv("SA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sa_cache")),
        "qg_pt_flan",
    )
    os.makedirs(cache_dir, exist_ok=True)

    # 1) Court-circuit: cache déjà complet -> ne rien faire
    if _has_required_files(cache_dir) and not _env_bool("PT_QG_FORCE_SYNC", False):
        log.info(f"[QG][s3] cache OK -> {cache_dir} (pas de sync)")
        return cache_dir

    # 2) Option: sauter la sync (dev)
    if _env_bool("PT_QG_SKIP_S3_SYNC", False):
        if _has_required_files(cache_dir):
            log.warning(f"[QG][s3] PT_QG_SKIP_S3_SYNC=1 et cache complet -> {cache_dir}")
            return cache_dir
        else:
            log.warning("[QG][s3] PT_QG_SKIP_S3_SYNC=1 mais cache incomplet.")
            return ""

    # 3) Sync S3 -> cache (ne télécharger que ce qui manque)
    ok = _sync_s3_dir_to_local(s3_prefix, cache_dir)
    print(f"téléchargement ou cache présent{ok}")

    # 4) Vérif finale: si les fichiers requis sont là, on considère OK
    if not _has_required_files(cache_dir):
        msg = (f"S3 sync KO ou incomplet vers {cache_dir}. "
               f"Requis: config.json, tokenizer.json|spiece.model, pytorch_model.bin|model.safetensors.")
        if _env_bool("REQUIRE_S3", False):
            raise RuntimeError(msg + " (REQUIRE_S3=1)")
        log.warning("[QG][s3] " + msg)
        return ""

    return cache_dir


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
        log.warning("[QG][s3] " + msg)
        return ""
    return cache_dir

# ───────────────────────────────────── Chargement modèle unique
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
        local_dir, dtype=dtype, device_map=dev_map, local_files_only=local_only
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
        model_id, dtype=dtype, device_map=dev_map, local_files_only=local_only
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
        log.error(f"[QG] ensure S3 failed: {e}")
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
        log.warning(f"[QG] primary load failed ({e}); fallback={fb}")
        model, tok = _load_qg_from_hf(fb)
        _QG_SOURCE = f"hf:{fb}"

    if enable_qa:
        try:
            dev = 0 if torch.cuda.is_available() else -1
            qa = hf_pipeline("question-answering", model="deepset/roberta-base-squad2", device=dev)
        except Exception as e:
            log.warning(f"[QG] QA filter not available: {e}")

    _QG = FlanT5QuestionGenerator(model, tok, qa_pipeline=qa, task_prefix="question: ")
    log.info(f"[QG] ready from {_QG_SOURCE} in {time.time()-t0:.2f}s")

# ───────────────────────────────────── Public dataclass
@dataclass
class QuestionsOutput:
    questions_fr: List[str]
    questions_en: List[str]
    model_used: str
    # spans utilisés pour debug/feedback ciblé
    internal_spans: List[str]

# ───────────────────────────────────── Génération standard (ton exemple)
def generate_questions_from_document(
    *,
    full_text_en: str,
    num_questions: int = 5,
    n_spans: int = 8,
    difficulty: str = "intermediate",
    style: str = "exam",
    diversity: float = 0.45,
    length: str = "medium",
) -> Tuple[List[str], List[str]]:
    _ensure_qg_ready(); assert _QG is not None
    spans = build_document_spans(full_text_en, n_spans=n_spans, window=1, max_span_len=5) or [full_text_en]
    # diversité : on mélange un peu l’ordre pour simuler la dispersion des spans (optionnel)
    if 0.0 <= diversity < 1.0 and len(spans) > 1:
        step = max(1, int(round((1.0 - diversity) * len(spans))))
        spans = spans[::step] + [s for i, s in enumerate(spans) if i % step != 0]
    qs_en = _QG.generate_from_spans(
        passages_en=spans,
        num_questions=max(1, int(num_questions)),
        difficulty=difficulty,
        style=style,
        length=length,
        force_in_context=True,
        use_qa_filter=False,
    )
    return qs_en, spans





# ───────────────────────────────────── Feedback → réécriture / régénération
def _map_payload_to_labels(payload: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    diff = (payload.get("difficulty") or "").lower()
    if diff in {"difficile", "hard", "advanced", "intermediate+"}:
        labels.append("increase_difficulty")
    elif diff in {"facile", "easy", "beginner"}:
        labels.append("decrease_difficulty")

    if (payload.get("avoid_trivial") or "").lower() in {"1","true","yes"}:
        labels.append("avoid_trivial")

    if payload.get("style"):
        labels.append("change_style")

    if payload.get("section"):
        labels.append("focus_section")

    n = int(payload.get("n", 0) or 0)
    if n and n <= 3:
        labels.append("length_shorter")
    elif n and n >= 8:
        labels.append("length_longer")

    return labels

# --- sentence split (NLTK sinon regex) ---
def _split_into_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    try:
        import nltk  # type: ignore
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                pass
        sents = nltk.sent_tokenize(t)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', t)
    sents = [re.sub(r"\s+", " ", s).strip() for s in sents if len(s.strip().split()) > 3]
    return sents


def _make_spans_from_sentences(sents: list[str], centers_idx: list[int], window: int = 1, max_span_len: int = 5) -> list[str]:
    spans = []
    for idx in centers_idx:
        start = max(0, idx - window)
        end   = min(len(sents), idx + window + 1)
        span_sents = sents[start:end]
        span = " ".join(span_sents[:max_span_len])
        spans.append(span)
    uniq, seen = [], set()
    for sp in spans:
        key = re.sub(r"\W+", "", sp.lower())
        if key not in seen:
            seen.add(key); uniq.append(sp)
    return uniq


def _select_diverse_sentences(sents: list[str], n_spans: int = 8, random_state: int = 42) -> list[int]:
    if len(sents) <= n_spans:
        return list(range(len(sents)))
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
        vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
        X = vec.fit_transform(sents)
        k = max(2, n_spans)
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        centers_idx = []
        centers = km.cluster_centers_
        labels = km.labels_
        for c in range(k):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            centroid = centers[c]
            subX = X[idxs]
            sims = subX @ centroid
            best_local = idxs[sims.argmax()]
            centers_idx.append(int(best_local))
        if len(centers_idx) < n_spans:
            remaining = [i for i in range(len(sents)) if i not in centers_idx]
            step = max(1, len(remaining) // (n_spans - len(centers_idx) + 1))
            centers_idx += remaining[::step][: (n_spans - len(centers_idx))]
        return sorted(list(dict.fromkeys(centers_idx)))[:n_spans]
    except Exception:
        # fallback simple et robuste si sklearn indispo
        step = max(1, len(sents) // n_spans)
        return list(range(0, len(sents), step))[:n_spans]


def build_document_spans(text_en: str, n_spans: int = 8, window: int = 1, max_span_len: int = 5) -> list[str]:
    sents = _split_into_sentences(text_en)
    if not sents:
        return []
    centers_idx = _select_diverse_sentences(sents, n_spans=n_spans)
    spans = _make_spans_from_sentences(sents, centers_idx, window=window, max_span_len=max_span_len)
    return spans


def regenerate_questions_with_feedback(
    *,
    full_text_en: str,
    payload: Optional[Dict[str, Any]] = None,
    default_num_questions: int = 5
) -> List[str]:
    payload = payload or {}
    _ensure_qg_ready(); assert _QG is not None

    # mapping robuste
    diff = (payload.get("difficulty") or "intermediate").strip().lower()
    if diff in {"facile","easy","beginner"}: difficulty = "easy"
    elif diff in {"intermediaire","intermédiaire","moyen","medium","intermediate"}: difficulty = "intermediate"
    elif diff in {"difficile","advanced","avancé","avance"}: difficulty = "advanced"
    else: difficulty = "intermediate"

    style = (payload.get("style") or "exam").strip().lower()
    if style not in {"academic","exam","concise","elaborated"}: style = "exam"

    length = (payload.get("length") or "medium").strip().lower()
    if length not in {"short","medium","long"}: length = "medium"

    raw_div = str(payload.get("diversity", "0.5")).replace(",", ".")
    try: diversity = float(raw_div)
    except Exception: diversity = 0.5
    diversity = max(0.0, min(1.0, diversity))

    want_n = max(1, min(10, int(payload.get("n", default_num_questions) or default_num_questions)))

    spans = select_top_spans(full_text_en, k=max(1, 8), diversity=diversity) or [full_text_en]

    qs_en = _QG.generate_from_spans(
        passages_en=spans,
        num_questions=want_n,
        difficulty=difficulty,
        style=style,
        length=length,
        use_qa_filter=_env_bool("ENABLE_QA_FILTER", False),
    )
    return postprocess_questions(qs_en)[:want_n]



# ───────────────────────────────────── Wrappers PDF (si utile côté API)
def run_questions_pipeline(pdf_bytes: bytes, num_questions: int = 5, doc_id: Optional[str] = None) -> QuestionsOutput:
    raw = utils.extract_text_safe(pdf_bytes)
    if not raw or not raw.strip():
        raise ValueError("Le PDF ne contient pas de texte exploitable.")
    cleaned = " ".join(raw.split())

    sum_cfg = _get_sum_cfg()
    if detect_lang_simple(cleaned) == "fr":
        english = translate_fr_to_en([cleaned], sum_cfg)[0]
    else:
        english = clean_text_en(cleaned)

    qs_en, spans = generate_questions_from_document(
        full_text_en=english,
        num_questions=5,  # première passe = 5 fixes
        n_spans=8,
        difficulty="intermediate",
        style="exam",
        diversity=0.45,
        length="medium",
    )

    # Utiliser le doc_id fourni par main.py pour stocker l'état
    if not doc_id:
        doc_id = _mk_doc_key(english)  # fallback si jamais
    _QG_SESS[doc_id] = QGConstraintsState(
        passages_en=spans,
        num_questions=5,
        difficulty="intermediate",
        style="exam",
        diversity=0.45,
        length="medium",
        force_in_context=True,
    )

    qs_fr = [translate_en_to_fr(q, sum_cfg) for q in qs_en]
    return QuestionsOutput(
        questions_fr=qs_fr,
        questions_en=qs_en,
        model_used=(_QG_SOURCE or "unknown"),
        internal_spans=spans,
    )


def run_questions_feedback_pipeline(pdf_bytes: bytes, payload: Optional[Dict[str, Any]] = None, default_num_questions: int = 5) -> QuestionsOutput:
    raw = utils.extract_text_safe(pdf_bytes)
    if not raw or not raw.strip():
        raise ValueError("Le PDF ne contient pas de texte exploitable.")
    cleaned = " ".join(raw.split())

    sum_cfg = _get_sum_cfg()
    english = translate_fr_to_en([cleaned], sum_cfg)[0] if detect_lang_simple(cleaned) == "fr" else clean_text_en(cleaned)

    payload = payload or {}
    doc_id = payload.get("doc_id")  # <- pas de re-hash ici
    base = _QG_SESS.get(doc_id)

    if base is None:
        # fallback si la 1ʳᵉ passe n'a pas pu enregistrer (rare)
        spans = build_document_spans(english, n_spans=8) or [english]
        base = QGConstraintsState(passages_en=spans, num_questions=default_num_questions)

    action = payload.get("action", "reject")        # 'accept' | 'reject' | 'update' | 'regenerate'
    reason = payload.get("reason")                  # ex. 'too-hard'
    fields = payload.get("fields") or {}            # overrides optionnels (ex. {"num_questions": 7})

    if action == "accept":
        newc = base
        regenerated = []
    else:
        if action == "reject":
            assert reason is not None, "Reject requires a reason"
        newc = apply_feedback_rules(base, reason, fields) if reason else base
        regenerated = _QG.generate_from_spans(
            passages_en=newc.passages_en,
            num_questions=newc.num_questions,
            difficulty=newc.difficulty,
            style=newc.style,
            length=newc.length,
            diversity=newc.diversity,  # <- IMPORTANT
            force_in_context=newc.force_in_context,
            use_qa_filter=False,
        )

    _QG_SESS[doc_id] = newc
    qs_en = regenerated
    qs_fr = [translate_en_to_fr(q, sum_cfg) for q in qs_en]

    log.info("[QG/FB] reason=%s | n=%d | diff=%s | style=%s | length=%s | diversity=%.2f | spans=%d",
             reason, newc.num_questions, newc.difficulty, newc.style, newc.length, newc.diversity,
             len(newc.passages_en))

    return QuestionsOutput(questions_fr=qs_fr, questions_en=qs_en, model_used=(_QG_SOURCE or "unknown"), internal_spans=newc.passages_en, doc_id=doc_id)

