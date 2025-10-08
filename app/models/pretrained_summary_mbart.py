
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_tok = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

FEEDBACK_TOKENS = {
    "length": {"court":"<LEN_SHORT>", "standard":"<LEN_MEDIUM>", "long":"<LEN_LONG>"},
    "tone": {"formel":"<TONE_FORMAL>", "neutre":"<TONE_NEUTRAL>", "informel":"<TONE_CASUAL>"},
    "focus": {"général":"<FOCUS_GENERAL>", "détails":"<FOCUS_DETAILS>", "résultats":"<FOCUS_RESULTS>","méthodes":"<FOCUS_METHODS>","limites":"<FOCUS_LIMITATIONS>","applications":"<FOCUS_APPLICATIONS>"},
    "structure": {"paragraphes":"<STRUCT_PARAGRAPHS>", "puces":"<STRUCT_BULLETS>", "sections":"<STRUCT_SECTIONS>"},
    "couverture": {"concis":"<COVER_KEYPOINTS>", "complet":"<COVER_COMPREHENSIVE>"},
    "style": {"abstractive":"<STYLE_ABSTRACTIVE>", "extractive":"<STYLE_EXTRACTIVE>"},
    "nombres": {"garder":"<NUM_KEEP>", "réduire":"<NUM_MINIMIZE>"},
    "citations": {"inclure":"<CITE_INCLUDE>", "exclure":"<CITE_EXCLUDE>"},
}

def setup(local_mbart_checkpoint_dir: str):
    global _tok, _model
    if _tok is None or _model is None:
        _tok = AutoTokenizer.from_pretrained(local_mbart_checkpoint_dir, use_fast=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(local_mbart_checkpoint_dir).to(_device).eval()

def _build_prefix(length="standard", ton="neutre", focus="général", structure="paragraphes", couverture="concis", style="abstractive", chiffres="garder", citations="inclure"):
    parts = [
        FEEDBACK_TOKENS["length"].get(length, "<LEN_MEDIUM>"),
        FEEDBACK_TOKENS["tone"].get(ton, "<TONE_NEUTRAL>"),
        FEEDBACK_TOKENS["focus"].get(focus, "<FOCUS_GENERAL>"),
        FEEDBACK_TOKENS["structure"].get(structure, "<STRUCT_PARAGRAPHS>"),
        FEEDBACK_TOKENS["couverture"].get(couverture, "<COVER_KEYPOINTS>"),
        FEEDBACK_TOKENS["style"].get(style, "<STYLE_ABSTRACTIVE>"),
        FEEDBACK_TOKENS["nombres"].get(chiffres, "<NUM_KEEP>"),
        FEEDBACK_TOKENS["citations"].get(citations, "<CITE_INCLUDE>"),
        "<DOC_START>",
    ]
    return " ".join(parts) + " "

@torch.no_grad()
def summarize(text_en: str, max_new_tokens: int = 280, beams: int = 4) -> str:
    inputs = _tok([text_en], return_tensors="pt", truncation=True).to(_model.device)
    out = _model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beams)
    return _tok.batch_decode(out, skip_special_tokens=True)[0].strip()

@torch.no_grad()
def summarize_with_feedback(text_en: str, feedback: Dict[str, str], max_new_tokens: int = 280, beams: int = 4) -> str:
    prefix = _build_prefix(**feedback)
    input_text = prefix + text_en
    inputs = _tok([input_text], return_tensors="pt", truncation=True).to(_model.device)
    out = _model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beams)
    return _tok.batch_decode(out, skip_special_tokens=True)[0].strip()
