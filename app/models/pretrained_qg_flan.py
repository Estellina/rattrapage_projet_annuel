
import re, torch
from typing import List, Dict, Any, Literal
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

STOP = set("a an the and or of to in for on with at from by as is are was were be being been it this that which who whose whom".split())

_tok = None
_model = None
_qa = None

def setup(model_id: str = "google/flan-t5-large", enable_qa: bool = False):
    global _tok, _model, _qa
    if _tok is None or _model is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        _tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto").eval()
    if enable_qa and _qa is None:
        dev = 0 if torch.cuda.is_available() else -1
        _qa = pipeline("question-answering", model="deepset/roberta-base-squad2", device=dev)

def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s for s in sents if len(s.split()) >= 6]

def sent_keywords(s: str):
    return re.findall(r"[A-Za-z][A-Za-z\-]{4,}", s.lower())

def select_top_spans(text: str, k: int = 5):
    sents = split_sentences(text)
    if not sents: return [text.strip()]
    def score(s):
        toks = [t for t in sent_keywords(s) if t not in STOP]
        return len(set(toks)) + 0.2*len(toks)
    ranked = sorted(sents, key=score, reverse=True)
    picked = []
    for s in ranked:
        if all(len(set(s.lower().split()) & set(p.lower().split()))/max(1,len(set(s.split()))) < 0.5 for p in picked):
            picked.append(s)
        if len(picked) >= k: break
    return picked[:k]

def force_words_ids_from_span(span: str, tok):
    kws = [w for w in sent_keywords(span) if w not in STOP]
    kws = sorted(set(kws), key=len, reverse=True)[:2]
    ids_list = []
    for w in kws:
        ids = tok.encode(w, add_special_tokens=False)
        if ids: ids_list.append(ids)
    return ids_list

def postprocess_questions(cands: List[str]) -> List[str]:
    out, seen = [], set()
    for q in cands:
        q = (q or "").strip()
        q = re.sub(r"^\s*(Q:|Question:)\s*", "", q, flags=re.I)
        if not q: continue
        if not q.endswith("?"): q = re.sub(r"[.]+$","", q).strip()+"?"
        q = q[0].upper()+q[1:] if q else q
        key = re.sub(r"\W+","", q.lower())
        if key not in seen:
            seen.add(key); out.append(q)
    return out

def qa_supported_long(context: str, question: str, min_score: float = 5.0, chunk_chars: int = 1600, overlap: int = 200) -> bool:
    if _qa is None:
        return True
    if len(context) <= chunk_chars:
        try:
            out = _qa(question=question, context=context)
            return out.get("score",0.0) >= min_score and bool(out.get("answer","").strip())
        except Exception:
            return True
    n, best, step = len(context), 0.0, max(1, chunk_chars - overlap)
    for start in range(0, n, step):
        piece = context[start:min(n, start+chunk_chars)]
        try:
            best = max(best, _qa(question=question, context=piece).get("score",0.0))
            if best >= min_score: return True
        except Exception:
            pass
        if start+step >= n: break
    return best >= min_score

@torch.no_grad()
def generate_questions(text_en: str, num_questions: int = 5, per_span_ret: int = 2, use_qa: bool = True, max_new_tokens: int = 64) -> List[str]:
    spans = select_top_spans(text_en, k=num_questions)
    all_q = []
    for sp in spans:
        prompt = ("Read the text and write one open-ended, content-specific question that refers to the key concepts.\n"
                  f"Text:\n{sp}\n"
                  "Question:")
        inputs = _tok([prompt], return_tensors="pt").to(_model.device)
        force_ids = force_words_ids_from_span(sp, _tok)
        gen = _model.generate(**inputs, num_beams=6, num_return_sequences=per_span_ret, max_new_tokens=max_new_tokens,
                              no_repeat_ngram_size=3, length_penalty=0.9, early_stopping=True,
                              force_words_ids=force_ids if force_ids else None)
        outs = _tok.batch_decode(gen, skip_special_tokens=True)
        outs = postprocess_questions(outs)
        if use_qa:
            outs = [q for q in outs if qa_supported_long(sp, q, min_score=5.0, chunk_chars=1200)]
        all_q.extend(outs)
    final = postprocess_questions(all_q)
    return final[:num_questions]

Difficulty = Literal["beginner","intermediate","advanced"]
def build_feedback_instruction(labels: List[str], payload: Dict[str,Any]):
    parts = []
    if "increase_difficulty" in labels: parts.append("make it more advanced and analytical")
    if "decrease_difficulty" in labels: parts.append("simplify language and reduce jargon")
    if "avoid_trivial" in labels or "too_trivial" in labels: parts.append("avoid 'what is' patterns; use why/how/under what conditions")
    if "change_style" in labels: parts.append(f"use a {payload.get('style','academic')} style")
    if "focus_section" in labels and payload.get("section"): parts.append(f"focus on '{payload['section']}'")
    if "length_shorter" in labels: parts.append("make it shorter")
    if "length_longer" in labels: parts.append("expand slightly")
    return "; ".join(parts) if parts else "improve clarity and specificity"

@torch.no_grad()
def apply_feedback(questions_en: List[str], source_text_en: str, labels: List[str], payload: Dict[str,Any] | None = None, max_new_tokens: int = 64) -> List[str]:
    payload = payload or {}
    instr = build_feedback_instruction(labels, payload)
    doc_kws = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{6,}", source_text_en.lower()) if w not in STOP]
    forced = []
    for w in list(dict.fromkeys(doc_kws))[:2]:
        ids = _tok.encode(w, add_special_tokens=False)
        if ids: forced.append(ids)
    forced = forced or None
    prompts = [f"Rewrite the question according to these instructions: {instr}.\nKeep it grounded in this text:\n{source_text_en}\nOriginal question: {q}\nRewritten question:" for q in questions_en]
    inputs = _tok(prompts, padding=True, truncation=True, return_tensors="pt").to(_model.device)
    gen = _model.generate(**inputs, num_beams=4, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3, length_penalty=1.0, force_words_ids=forced)
    outs = _tok.batch_decode(gen, skip_special_tokens=True)
    outs = postprocess_questions(outs)
    outs = [q for q in outs if qa_supported_long(source_text_en, q, min_score=5.0, chunk_chars=1600)]
    return outs
