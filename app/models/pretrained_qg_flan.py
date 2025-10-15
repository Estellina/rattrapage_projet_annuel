# app/models/pretrained_qg_flan.py
# ---------------------------------------------------------------------
# Helpers + classe de génération de questions pour modèles T5/FLAN.
# Le chargement (model/tokenizer/qa) est effectué ailleurs (pipeline).
# ---------------------------------------------------------------------

from __future__ import annotations
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

# ------------------------------- Utils texte -------------------------------

STOP: set[str] = set(
    """
    a an the and or of to in for on with at from by as is are was were be being been it
    this that these those which who whose whom where when while than then thus hence
    into onto over under between within without about across against among around
    can could may might must shall should will would do does did done
    we you they he she i their your our his her its them him her us me
    """.split()
)

_SENT_SPLIT = re.compile(r"(?<=[\.!\?])\s+")


# -------------------- Constraints (inspiré du notebook) --------------------
class QGConstraints:
    def __init__(self, *, difficulty="intermediate", style="exam", length="medium"):
        self.difficulty = difficulty
        self.style = style
        self.length = length

    def max_new_tokens(self) -> int:
        return {"short": 32, "medium": 64, "long": 96}.get(self.length, 64)


    def length_to_max_new(self) -> int:
        return {"short": 32, "medium": 64, "long": 96}.get(self.length, 64)

    def rules_text(self) -> str:
        rules = []
        if self.difficulty == "easy":
            rules += [
                "Prefer recall/definition; avoid multi-hop reasoning.",
                "Keep each question concise (<= 18 words).",
            ]
        elif self.difficulty == "intermediate":
            rules += [
                "Target comprehension/application; allow a modest level of inference.",
                "Avoid pure definitions unless necessary.",
            ]
        elif self.difficulty == "advanced":
            rules += [
                "Prefer analysis/evaluation; compare concepts or discuss assumptions/limitations.",
                "Encourage critical thinking and multi-step reasoning.",
            ]
        if self.style == "concise":
            rules.append("Write succinctly.")
        elif self.style == "elaborated":
            rules.append("Allow slightly longer, well-structured questions.")
        elif self.style == "exam":
            rules.append("Use exam-like phrasing appropriate for assessment.")
        # longueur
        if self.length == "short":
            rules.append("Keep it short (<= 12-18 words).")
        elif self.length == "long":
            rules.append("Allow a more elaborate phrasing (up to 25-30 words).")
        return ("\n- " + "\n- ".join(rules)) if rules else ""



def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def split_sentences(text: str, min_words: int = 6) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    cand = _SENT_SPLIT.split(text)
    out = [normalize_ws(c) for c in cand if len(c.split()) >= min_words]
    return out or [normalize_ws(text)]

def keyword_tokens(s: str, min_len: int = 5) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", (s or "").lower())
    return [t for t in toks if len(t) >= min_len and t not in STOP]

# ------------------------- Sélection de spans (Top-K) -----------------------

def select_top_spans(
    text: str,
    k: int = 5,
    min_words: int = 6,
    diversity: float = 0.5
) -> List[str]:
    """
    Heuristique simple: score = diversité de tokens + densité de mots-clés.
    On impose une diversité entre spans sélectionnés (Jaccard < threshold).
    """
    sents = split_sentences(text, min_words=min_words)
    if not sents:
        return []

    def score(s: str) -> float:
        toks = keyword_tokens(s)
        return len(set(toks)) + 0.2 * len(toks)

    ranked = sorted(sents, key=score, reverse=True)
    picked: List[str] = []
    picked_sets: List[set[str]] = []

    for s in ranked:
        ks = set(keyword_tokens(s))
        if not ks:
            continue
        ok = True
        for existing in picked_sets:
            inter = len(ks & existing)
            jac = inter / max(1, len(ks | existing))
            if jac > (1 - diversity):  # plus de diversité => jac faible
                ok = False
                break
        if ok:
            picked.append(s)
            picked_sets.append(ks)
        if len(picked) >= k:
            break

    return picked[:k] or sents[:k]

# ----------------------- Force-words pour T5/FLAN --------------------------

def force_words_ids_for_span(
    span: str,
    tokenizer: PreTrainedTokenizerBase,
    max_terms: int = 2
) -> Optional[List[List[int]]]:
    """
    Construit une petite liste d'IDs à forcer (termes saillants du span).
    Transformers attend List[List[int]]; chaque sous-liste = un terme.
    """
    terms = sorted(set(keyword_tokens(span)), key=len, reverse=True)[:max_terms]
    ids_list: List[List[int]] = []
    for w in terms:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            ids_list.append(ids)
    return ids_list or None

# ------------------------- Post-traitement des questions --------------------

_Q_PREFIX = re.compile(r"^\s*(Q:|Question:)\s*", re.I)


# ----------------------------- QA filter optionnel -------------------------
# --- RÈGLES & PROMPTS GUIDÉS ---
def _fix_digit_intrusions(s: str) -> str:
    return re.sub(r'(?<=[a-z])2(?=[a-z])', 'ti', s)

def postprocess_questions(cands: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for q in cands:
        q = (q or "").strip()
        if not q:
            continue
        q = q.splitlines()[0].strip()
        q = _fix_digit_intrusions(q)
        q = re.sub(r"[. ]+$", "", q)
        if not q.endswith("?"):
            q += "?"
        if q:
            q = q[0].upper() + q[1:]
        key = re.sub(r"\W+", "", q.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out










def qa_supported_long(
    context: str,
    question: str,
    qa_pipeline: Any = None,
    min_score: float = 5.0,
    chunk_chars: int = 1600,
    overlap: int = 200
) -> bool:
    """
    Si un pipeline QA (SQuAD) est fourni, on vérifie qu'il existe une réponse plausible.
    Fonctionne sur contextes longs par fenêtrage.
    """
    if qa_pipeline is None:
        return True
    ctx = (context or "").strip()
    if not ctx:
        return True

    n = len(ctx)
    step = max(1, chunk_chars - overlap)

    if n <= chunk_chars:
        try:
            out = qa_pipeline(question=question, context=ctx)
            return bool(out.get("answer", "").strip()) and out.get("score", 0.0) >= min_score
        except Exception:
            return True

    best = 0.0
    for start in range(0, n, step):
        window = ctx[start: min(n, start + chunk_chars)]
        try:
            out = qa_pipeline(question=question, context=window)
            best = max(best, out.get("score", 0.0))
            if best >= min_score:
                return True
        except Exception:
            pass
        if start + step >= n:
            break
    return best >= min_score

# ------------------------- ENV helpers (génération) ------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default).lower()) in {"1", "true", "yes"}

# --------------------- Classe principale de génération ---------------------

# Ajoute ce helper dans le fichier (par ex. sous les _env_* helpers)
def _force_words_allowed(model) -> bool:
    """
    On n'utilise force_words_ids que si:
      - autorisé par l'ENV (QG_ENABLE_FORCE_WORDS=true) ET
      - le modèle expose bien la méthode 'constrained_beam_search'
    Sinon on évite le mode 'constrained' qui plante sur certaines versions HF.
    """
    from inspect import ismethod
    if os.getenv("QG_ENABLE_FORCE_WORDS", "false").lower() not in {"1", "true", "yes"}:
        return False
    return hasattr(model, "constrained_beam_search") and callable(getattr(model, "constrained_beam_search", None))

def _extra_rules(difficulty: str, style: str) -> str:
    rules = []
    if difficulty == "easy":
        rules += [
            "Prefer recall/definition; avoid multi-hop reasoning.",
            "Keep each question concise (<= 18 words).",
        ]
    elif difficulty == "intermediate":
        rules += [
            "Target comprehension/application; allow a modest level of inference.",
            "Avoid pure definitions unless necessary.",
        ]
    elif difficulty == "advanced":
        rules += [
            "Prefer analysis/evaluation; compare concepts or discuss assumptions/limitations.",
            "Encourage critical thinking and multi-step reasoning.",
        ]
    if style == "concise":
        rules.append("Write succinctly.")
    elif style == "elaborated":
        rules.append("Allow slightly longer, well-structured questions.")
    elif style == "exam":
        rules.append("Use exam-like phrasing appropriate for assessment.")
    return ("\n- " + "\n- ".join(rules)) if rules else ""


def build_prompt(passages_en: list[str], N: int, difficulty: str, style: str,
                 force_in_context: bool = True) -> str:
    joined = "\n\n".join(f"[Span {i+1}]\n{p.strip()}" for i, p in enumerate(passages_en))
    strict = "- Stay strictly within the provided passage(s); avoid external knowledge.\n" if force_in_context else ""
    return (
        "You are a question generation model.\n\n"
        "Constraints:\n"
        f"- Difficulty: {difficulty}\n"
        f"- Style: {style}\n"
        f"- Number of questions: {N}\n"
        f"{strict}"
        f"{_extra_rules(difficulty, style)}\n\n"
        "Passage(s):\n"
        f"{joined}\n\n"
        f"Generate {N} open-ended question(s) in English. One idea per question.\n"
        "Output format: one question per line, no numbering, no bullets, no extra text."
    )


class FlanT5QuestionGenerator:
    def __init__(self, model, tok, qa_pipeline=None, task_prefix="question: "):
        self.model = model
        self.tok = tok
        self.qa = qa_pipeline
        self.task_prefix = task_prefix

    # --------- Génération de questions à partir d'un texte (EN) ----------

    # Remplacer TOUT le body de FlanT5QuestionGenerator.generate par ceci
    @torch.no_grad()
    def generate(
            self,
            text_en: str,
            *,
            num_questions: int = 5,
            per_span_ret: int = 2,
            max_new_tokens: int = 64,
            min_words_per_sentence: int = 6,
            diversity: float = 0.5,
            use_qa_filter: bool = True
    ) -> List[str]:
        num_beams = _env_int("QG_GEN_NUM_BEAMS", 4)
        no_repeat = _env_int("QG_GEN_NO_REPEAT_NGRAM_SIZE", 4)
        rep_pen = _env_float("QG_GEN_REPETITION_PENALTY", 1.15)
        do_sample = _env_bool("QG_GEN_DO_SAMPLE", False)
        length_pen = _env_float("QG_GEN_LENGTH_PENALTY", 0.9)

        spans = [text_en]  # ici, on génère sur la portion passée (côté pipeline on split déjà)
        device = getattr(self.model, "device", torch.device("cpu"))
        all_q: List[str] = []

        for sp in spans:
            prompt = (
                    f"{self.task_prefix}".strip()
                    + (
                        " Read the text and write one open-ended, content-specific question that refers to the key concepts.\n"
                        f"Text:\n{sp}\n"
                        "Question:"
                    )
            ).strip()

            inputs = self.tok([prompt], return_tensors="pt", truncation=True).to(device)
            want_force = _force_words_allowed(self.model)
            force_ids = force_words_ids_for_span(sp, self.tok, max_terms=2) if want_force else None

            kwargs = dict(
                num_beams=num_beams,
                num_return_sequences=per_span_ret,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=no_repeat,
                length_penalty=length_pen,
                early_stopping=True,
                repetition_penalty=rep_pen,
                do_sample=do_sample,
            )

            try:
                gen = self.model.generate(**inputs, force_words_ids=force_ids, **kwargs)
            except Exception:
                # Fallback sans contraintes (évite le mode constrained-beam-search)
                gen = self.model.generate(**inputs, force_words_ids=None, **kwargs)

            outs = self.tok.batch_decode(gen, skip_special_tokens=True)
            outs = postprocess_questions(outs)
            if use_qa_filter:
                outs = [q for q in outs if qa_supported_long(sp, q, self.qa, min_score=5.0, chunk_chars=1200)]
            all_q.extend(outs)

        final = postprocess_questions(all_q)
        return final[:num_questions]

    @torch.no_grad()
    def generate_from_spans(
            self,
            passages_en: List[str],
            *,
            num_questions: int = 5,
            difficulty: str = "intermediate",
            style: str = "exam",
            length: str = "medium",
            diversity: float = 0.45,
            force_in_context: bool = True,
            use_qa_filter: bool = False,
    ) -> List[str]:
        import os, re, torch
        N = max(1, int(num_questions))

        # Longueur cible
        max_new = {"short": 48, "medium": 64, "long": 96}.get(length, 64)

        # Décodage piloté par la diversité
        if diversity <= 0.10:
            # stable
            do_sample = False
            num_beams = 4
            sample_kwargs = {}
        else:
            # varié
            do_sample = True
            num_beams = 1
            top_p = min(0.98, 0.80 + diversity * 0.18)  # 0.80..0.98
            temperature = min(1.30, 0.70 + diversity * 0.60)  # 0.70..1.30
            top_k = int(20 + diversity * 40)  # 20..60
            sample_kwargs = dict(do_sample=True, top_p=top_p, top_k=top_k, temperature=temperature)

        pad_id = getattr(self.tok, "pad_token_id", None) or getattr(self.tok, "eos_token_id", None) or 1
        eos_id = getattr(self.tok, "eos_token_id", None) or pad_id

        outs: List[str] = []
        if not passages_en:
            passages_en = [""]

        for i in range(N):
            # Reseed global RNG si sampling (évite les sorties figées)
            if do_sample:
                seed = int.from_bytes(os.urandom(8), "big")
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            span = passages_en[i % len(passages_en)]
            prompt = build_prompt([span], 1, difficulty, style, force_in_context=force_in_context)

            enc = self.tok(
                f"{self.task_prefix}{prompt}".strip(),
                return_tensors="pt",
                truncation=True,
                max_length=768,
            ).to(self.model.device)

            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                min_new_tokens=8,
                num_return_sequences=1,  # strict-N
                num_beams=num_beams,
                do_sample=do_sample,
                no_repeat_ngram_size=4,
                repetition_penalty=1.12,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **sample_kwargs
            )
            text = self.tok.decode(gen[0], skip_special_tokens=True).strip()
            outs.append(text)

        # Nettoyage / dédup / strict-N
        qs = postprocess_questions(outs)
        uniq, seen = [], set()
        for q in qs:
            k = re.sub(r"\W+", "", q.lower())
            if k in seen:
                continue
            seen.add(k);
            uniq.append(q)
            if len(uniq) >= N:
                break
        while len(uniq) < N:
            uniq.append(uniq[-1] if uniq else "What is the key idea?")
        return uniq[:N]

    # -------------------- Réécriture selon feedback ----------------------

    # Remplacer TOUT le body de FlanT5QuestionGenerator.rewrite_with_feedback par ceci
    @torch.no_grad()
    def rewrite_with_feedback(
            self,
            questions_en: Sequence[str],
            source_text_en: str,
            labels: Sequence[str],
            payload: Optional[Dict[str, Any]] = None,
            *,
            max_new_tokens: int = 64
    ) -> List[str]:
        payload = payload or {}
        instr = build_feedback_instruction(labels, payload)

        doc_kws = [w for w in keyword_tokens(source_text_en, min_len=6) if w not in STOP]
        forced: Optional[List[List[int]]] = []
        for w in list(dict.fromkeys(doc_kws))[:2]:
            ids = self.tok.encode(w, add_special_tokens=False)
            if ids:
                forced.append(ids)
        want_force = _force_words_allowed(self.model)
        if not forced or not want_force:
            forced = None

        prompts = [
            (
                    f"{self.task_prefix}".strip()
                    + (
                        f" Rewrite the question according to these instructions: {instr}.\n"
                        f"Keep it grounded in this text:\n{source_text_en}\n"
                        f"Original question: {q}\n"
                        f"Rewritten question:"
                    )
            ).strip()
            for q in questions_en
        ]

        inputs = self.tok(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(getattr(self.model, "device", torch.device("cpu")))

        kwargs = dict(
            num_beams=_env_int("QG_GEN_NUM_BEAMS", 4),
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=_env_int("QG_GEN_NO_REPEAT_NGRAM_SIZE", 3),
            length_penalty=_env_float("QG_GEN_LENGTH_PENALTY", 1.0),
            repetition_penalty=_env_float("QG_GEN_REPETITION_PENALTY", 1.15),
            do_sample=_env_bool("QG_GEN_DO_SAMPLE", False),
        )

        try:
            gen = self.model.generate(**inputs, force_words_ids=forced, **kwargs)
        except Exception:
            gen = self.model.generate(**inputs, force_words_ids=None, **kwargs)

        outs = self.tok.batch_decode(gen, skip_special_tokens=True)
        outs = postprocess_questions(outs)
        outs = [q for q in outs if qa_supported_long(source_text_en, q, self.qa, min_score=5.0, chunk_chars=1600)]
        return outs


# ---------------------- Construction d'instructions (feedback) ----------------------

Difficulty = Literal["beginner", "intermediate", "advanced"]

def build_feedback_instruction(labels: Sequence[str], payload: Dict[str, Any]) -> str:
    """
    Construit une instruction compacte à partir des labels UI.
    """
    labels = set([str(x).lower() for x in labels])
    p = []
    if "increase_difficulty" in labels: p.append("make it more advanced and analytical")
    if "decrease_difficulty" in labels: p.append("simplify language and reduce jargon")
    if "avoid_trivial" in labels or "too_trivial" in labels: p.append("avoid 'what is' patterns; prefer why/how/under what conditions")
    if "change_style" in labels and payload.get("style"): p.append(f"use a {payload['style']} style")
    if "focus_section" in labels and payload.get("section"): p.append(f"focus on '{payload['section']}'")
    if "length_shorter" in labels: p.append("make it shorter")
    if "length_longer" in labels: p.append("expand slightly")
    return "; ".join(p) if p else "improve clarity and specificity"

# ------------------------------ Public API ---------------------------------

__all__ = [
    "FlanT5QuestionGenerator",
    "select_top_spans",
    "postprocess_questions",
    "force_words_ids_for_span",
    "qa_supported_long",
    "build_feedback_instruction",
]
