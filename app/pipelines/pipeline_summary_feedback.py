
from ..config import cfg
from .. import utils
from ..s3_utils import download_s3_prefix
from ..models import pretrained_summary_mbart as pt_mbart
import pathlib
_PT_READY=False; _PT_DIR=None

def _ensure_pt():
    global _PT_READY, _PT_DIR
    if _PT_READY: return
    cache = pathlib.Path(cfg.CACHE_DIR) / "summary_pt_mbart"
    if cfg.PT_SUMMARY_S3_PREFIX:
        download_s3_prefix(cfg.PT_SUMMARY_S3_PREFIX, cache)
        _PT_DIR = str(cache)
    else:
        _PT_DIR = str(cache)
    pt_mbart.setup(_PT_DIR); _PT_READY=True

def run_summary_feedback_pipeline(source_text_fr: str, feedback: dict) -> str:
    _ensure_pt()
    text_en, _ = utils.maybe_translate(source_text_fr, target_lang="en", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
    out_en = pt_mbart.summarize_with_feedback(text_en, feedback)
    out_fr, _ = utils.maybe_translate(out_en, target_lang="fr", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
    return out_fr
