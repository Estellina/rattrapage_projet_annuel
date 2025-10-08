
from ..config import cfg
from .. import utils
from ..models import pretrained_qg_flan as pt_flan
_PT_READY=False
def _ensure_pt():
    global _PT_READY
    if _PT_READY: return
    pt_flan.setup(cfg.PT_QG_MODEL, enable_qa=cfg.ENABLE_QA_FILTER); _PT_READY=True

def run_questions_feedback_pipeline(source_text_fr: str, questions_fr: list[str], labels: list[str], payload: dict | None = None) -> list[str]:
    _ensure_pt()
    payload = payload or {}
    source_en, _ = utils.maybe_translate(source_text_fr, target_lang="en", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)
    qs_en = [utils.maybe_translate(q, target_lang="en", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)[0] for q in questions_fr]
    out_en = pt_flan.apply_feedback(qs_en, source_en, labels, payload)
    out_fr = [utils.maybe_translate(q, target_lang="fr", enable_offline=cfg.ENABLE_OFFLINE_TRANSLATION)[0] for q in out_en]
    return out_fr
