
import io, re
from typing import Tuple
from langdetect import detect
from pypdf import PdfReader

_WS = re.compile(r"[\t\r\f]+")
_MULTI_NL = re.compile(r"\n{3,}")
_BAD = re.compile(r"[\u200b\u200c\u200d\ufeff]")

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

def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _BAD.sub("", text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()

_translators = {}
def maybe_translate(text: str, target_lang: str = "en", enable_offline: bool = False) -> Tuple[str, str]:
    try:
        lang = detect(text) if text.strip() else target_lang
    except Exception:
        lang = target_lang
    if not enable_offline or lang.lower().startswith(target_lang.lower()):
        return text, lang
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    if lang.startswith("fr") and target_lang.startswith("en"):
        model_id = "Helsinki-NLP/opus-mt-fr-en"
    elif lang.startswith("en") and target_lang.startswith("fr"):
        model_id = "Helsinki-NLP/opus-mt-en-fr"
    else:
        model_id = "Helsinki-NLP/opus-mt-mul-en" if target_lang.startswith("en") else "Helsinki-NLP/opus-mt-en-mul"
    tok = _translators.get((lang,target_lang,"tok"))
    mdl = _translators.get((lang,target_lang,"mdl"))
    if tok is None or mdl is None:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _translators[(lang,target_lang,"tok")] = tok
        _translators[(lang,target_lang,"mdl")] = mdl
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    out = mdl.generate(**inputs, max_new_tokens=1024)
    trans = tok.batch_decode(out, skip_special_tokens=True)[0]
    return trans, lang

def simple_quality_score(source: str, generated: str) -> float:
    if not generated.strip():
        return 0.0
    s_words = set(re.findall(r"[A-Za-z][A-Za-z\-]{3,}", source.lower()))
    g_words = set(re.findall(r"[A-Za-z][A-Za-z\-]{3,}", generated.lower()))
    if not s_words or not g_words:
        return 0.2
    overlap = len(s_words & g_words) / max(1, len(g_words))
    length_pen = 0.0
    L = len(generated.split())
    if L < 40: length_pen += 0.2
    if L > 380: length_pen += 0.1
    return max(0.0, min(1.0, 0.6*overlap + (1.0 - length_pen)))
