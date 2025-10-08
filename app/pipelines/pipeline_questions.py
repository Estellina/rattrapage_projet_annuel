
from dataclasses import dataclass
from ..config import cfg
from .. import utils
from ..s3_utils import download_s3_file, download_s3_prefix
from ..models import fs_qg_loader as fs_qg
from ..models import pretrained_qg_flan as pt_flan
import pathlib

_FS_READY=False; _PT_READY=False

def _ensure_fs():
    global _FS_READY
    if _FS_READY: return
    cache = pathlib.Path(cfg.CACHE_DIR) / "qg_fs"
    ckpt = cache / "ckpt.pt"; tok_dir = cache / "tokenizer"
    if cfg.FS_QG_CKPT: download_s3_file(cfg.FS_QG_CKPT, ckpt)
    if cfg.FS_QG_TOK: download_s3_prefix(cfg.FS_QG_TOK, tok_dir)
    fs_qg.load_qg_model(str(ckpt), str(tok_dir)); _FS_READY=True

def _ensure_pt():
    global _PT_READY
    if _PT_READY: return
    pt_flan.setup(cfg.PT_QG_MODEL, enable_qa=cfg.ENABLE_QA_FILTER); _PT_READY=True

@dataclass
class QuestionsOutput:
    questions_fr: list
    questions_en: list
    model_used: str

def run_questions_pipeline(pdf_bytes: bytes, num_questions: int = 5) -> QuestionsOutput:
    raw = utils.extract_text_from_pdf(pdf_bytes)
    cleaned = utils.clean_text(raw)
    english, _ = utils.maybe_translate(cleaned, target_lang="en", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
    try:
        _ensure_fs()
        outs_en = fs_qg.generate_questions(english, n=num_questions, difficulty="auto", scope="auto", section=None)
        got, seen = [], set()
        for q in outs_en:
            key = "".join(ch for ch in q.lower() if ch.isalnum())
            if key not in seen and len(q) > 6:
                seen.add(key); got.append(q)
        if len(got) < max(3, num_questions//2): raise RuntimeError("too few")
        outs_fr = [utils.maybe_translate(q, target_lang="fr", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)[0] for q in got]
        return QuestionsOutput(questions_fr=outs_fr[:num_questions], questions_en=got[:num_questions], model_used="fs_qg")
    except Exception:
        _ensure_pt()
        outs_en = pt_flan.generate_questions(english, num_questions=num_questions, per_span_ret=2, use_qa=cfg.ENABLE_QA_FILTER)
        outs_fr = [utils.maybe_translate(q, target_lang="fr", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)[0] for q in outs_en]
        return QuestionsOutput(questions_fr=outs_fr, questions_en=outs_en, model_used="pt_flan")
