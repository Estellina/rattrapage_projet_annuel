
from dataclasses import dataclass
from ..config import cfg
from .. import utils
from ..s3_utils import download_s3_file, download_s3_prefix
from ..models import fs_summary_loader as fs_summary
from ..models import pretrained_summary_mbart as pt_mbart
import pathlib

_FS_READY=False; _PT_READY=False; _PT_DIR=None

def _ensure_fs():
    global _FS_READY
    if _FS_READY: return
    cache = pathlib.Path(cfg.CACHE_DIR) / "summary_fs"
    ckpt = cache / "ckpt.pth"
    tok_dir = cache / "tokenizer"
    if cfg.FS_SUMMARY_CKPT: download_s3_file(cfg.FS_SUMMARY_CKPT, ckpt)
    if cfg.FS_SUMMARY_TOK: download_s3_prefix(cfg.FS_SUMMARY_TOK, tok_dir)
    fs_summary.load_summary_model(str(ckpt), str(tok_dir))
    _FS_READY=True

def _ensure_pt():
    global _PT_READY, _PT_DIR
    if _PT_READY: return
    cache = pathlib.Path(cfg.CACHE_DIR) / "summary_pt_mbart"
    if cfg.PT_SUMMARY_S3_PREFIX:
        download_s3_prefix(cfg.PT_SUMMARY_S3_PREFIX, cache)
        _PT_DIR = str(cache)
    else:
        _PT_DIR = str(cache)
    pt_mbart.setup(_PT_DIR)
    _PT_READY=True

@dataclass
class SummaryOutput:
    text_fr: str
    text_en: str
    model_used: str
    quality: float

def run_summary_pipeline(pdf_bytes: bytes) -> SummaryOutput:
    raw = utils.extract_text_from_pdf(pdf_bytes)
    cleaned = utils.clean_text(raw)
    english, _ = utils.maybe_translate(cleaned, target_lang=cfg.TRANSLATION_TGT, enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
    try:
        _ensure_fs()
        src, sm_t, sm_s = fs_summary.pack_from_raw_text(english, fs_summary._tokenizer, fs_summary.PAD or 0)
        text_en = fs_summary.generate_grounded(fs_summary._model, src, sm_t, sm_s, fs_summary.BOS, fs_summary.EOS, fs_summary.PAD, fs_summary._tokenizer)
        quality = utils.simple_quality_score(english, text_en)
        if quality < 0.45:
            raise RuntimeError("low quality")
        text_fr, _ = utils.maybe_translate(text_en, target_lang="fr", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
        return SummaryOutput(text_fr=text_fr, text_en=text_en, model_used="fs_hibert", quality=quality)
    except Exception:
        _ensure_pt()
        text_en = pt_mbart.summarize(english)
        quality = utils.simple_quality_score(english, text_en)
        text_fr, _ = utils.maybe_translate(text_en, target_lang="fr", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
        return SummaryOutput(text_fr=text_fr, text_en=text_en, model_used="pt_mbart", quality=quality)
