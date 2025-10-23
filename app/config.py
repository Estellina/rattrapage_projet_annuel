# app/config.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
from pathlib import Path

# 1) Charger .env de façon robuste (racine + cwd)
try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[1]
    for cand in (ROOT/".env", ROOT/".env.local", Path.cwd()/".env"):
        if cand.exists():
            load_dotenv(dotenv_path=cand, override=False)
except Exception:
    pass


def _to_dir(uri: str | None) -> str:
    if not uri:
        return ""
    if uri.startswith("s3://"):
        return uri if uri.endswith("/") else uri.rsplit("/", 1)[0] + "/"
    return uri if uri.endswith("/") else uri + "/"

# Alias S3 : prendre d'abord PT_SUMMARY_S3_PREFIX, sinon PT_SUMMARY_S3_CHECKPOINT
_pt = os.getenv("PT_SUMMARY_S3_PREFIX") or os.getenv("PT_SUMMARY_S3_CHECKPOINT") or ""
if _pt:
    _pt = _to_dir(_pt)
    os.environ.setdefault("PT_SUMMARY_S3_PREFIX", _pt)


_cache = os.getenv("SA_CACHE_DIR") or os.getenv("PT_SUMMARY_CACHE_DIR") or os.getenv("CACHE_DIR")
if _cache:
    os.environ.setdefault("SA_CACHE_DIR", _cache)


# Caches/vars HF
_HF_CACHE = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or "/tmp/transformers_cache"
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_CACHE
Path(_HF_CACHE).mkdir(parents=True, exist_ok=True)

# Threads BLAS (safe)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@dataclass
class Config:
    # App
    APP_NAME: str = os.getenv("APP_NAME", "PDF Summarizer")
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}
    APP_SECRET: str = os.getenv("APP_SECRET", "change-me")

    # Infra / S3
    AWS_REGION: str = os.getenv("AWS_REGION", "eu-west-3")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "smart-assistant-bucket")
    S3_PREFIX: str = _to_dir(os.getenv("S3_PREFIX", ""))

    # Téléchargement modèles au démarrage
    PREWARM_MODELS: bool = os.getenv("PREWARM_MODELS", "true").lower() in {"1", "true", "yes"}

    # Cache local
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/sa_cache")

    # ── From-scratch Summary (HIBERT) — **depuis .env**
    FS_SUMMARY_CKPT: str = os.getenv("SUM_FS_CHECKPOINT_S3", "")  # s3://.../hibert_dapt_best.pth ou local
    # soit un dossier complet:
    FS_SUMMARY_TOK_DIR: str = _to_dir(os.getenv("SUM_FS_TOKENIZER_DIR", ""))
    # soit deux fichiers individuels:
    FS_SUMMARY_TOK_MERGES: str = os.getenv("SUM_FS_TOKENIZER_MERGES_S3", "")
    FS_SUMMARY_TOK_VOCAB: str  = os.getenv("SUM_FS_TOKENIZER_VOCAB_S3", "")

    # ── mBART checkpoint (local/S3) + fallback HF
    PT_SUMMARY_S3_PREFIX: str = _to_dir(os.getenv(
        "PT_SUMMARY_S3_CHECKPOINT",
        "s3://smart-assistant-bucket/models/summary/pretrained_summary/better_checkpoint/checkpoint_improved/checkpoint-25366/"
    ))
    PT_SUMMARY_HF_MODEL: str = os.getenv("PT_SUMMARY_HF_MODEL", "facebook/mbart-large-50")

    # ── Traduction FR->EN offline
    ENABLE_OFFLINE_TRANSLATION: bool = os.getenv("ENABLE_OFFLINE_TRANSLATION", "true").lower() in {"1", "true", "yes"}
    TRANSLATION_TGT: str = os.getenv("TRANSLATION_TGT", "en")

    # ── Garde-fous & tailles
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "30"))
    SUMMARY_MAX_CHARS: int = int(os.getenv("SUMMARY_MAX_CHARS", "20000"))

    # ── Chunking Summary (PT)
    SUMMARY_PT_MAX_TOKENS: int = int(os.getenv("SUMMARY_PT_MAX_TOKENS", "900"))
    SUMMARY_PT_OVERLAP: int    = int(os.getenv("SUMMARY_PT_OVERLAP", "150"))

    # ── Qualité
    SUMMARY_FEEDBACK_MIN_QUALITY: float = float(os.getenv("SUMMARY_FEEDBACK_MIN_QUALITY", "0.45"))
    # fallback sans tokenizer pour certaines variantes legacy
    SUMMARY_FEEDBACK_BIN_CHARS: int = int(os.getenv("SUMMARY_FEEDBACK_BIN_CHARS", "4000"))

    # Génération PT (mBART)
    SUMMARY_GEN_NUM_BEAMS: int = int(os.getenv("SUMMARY_GEN_NUM_BEAMS", "4"))
    SUMMARY_GEN_NUM_BEAM_GROUPS: int = int(os.getenv("SUMMARY_GEN_NUM_BEAM_GROUPS", "2"))
    SUMMARY_GEN_DIVERSITY_PENALTY: float = float(os.getenv("SUMMARY_GEN_DIVERSITY_PENALTY", "0.3"))
    SUMMARY_GEN_NO_REPEAT_NGRAM_SIZE: int = int(os.getenv("SUMMARY_GEN_NO_REPEAT_NGRAM_SIZE", "4"))
    SUMMARY_GEN_ENCODER_NO_REPEAT_NGRAM_SIZE: int = int(os.getenv("SUMMARY_GEN_ENCODER_NO_REPEAT_NGRAM_SIZE", "3"))
    SUMMARY_GEN_REPETITION_PENALTY: float = float(os.getenv("SUMMARY_GEN_REPETITION_PENALTY", "1.25"))
    SUMMARY_GEN_LENGTH_PENALTY: float = float(os.getenv("SUMMARY_GEN_LENGTH_PENALTY", "1.05"))
    SUMMARY_GEN_MIN_NEW_TOKENS: int = int(os.getenv("SUMMARY_GEN_MIN_NEW_TOKENS", "60"))
    SUMMARY_GEN_MAX_NEW_TOKENS: int = int(os.getenv("SUMMARY_GEN_MAX_NEW_TOKENS", "224"))
    SUMMARY_GEN_DO_SAMPLE: bool = os.getenv("SUMMARY_GEN_DO_SAMPLE", "false").lower() in {"1", "true", "yes"}

    # Traduction
    TRANSLATION_CHUNK_TOKENS: int = int(os.getenv("TRANSLATION_CHUNK_TOKENS", "480"))
    TRANSLATION_NUM_BEAMS: int = int(os.getenv("TRANSLATION_NUM_BEAMS", "4"))
    TRANSLATION_MAX_NEW: int = int(os.getenv("TRANSLATION_MAX_NEW", "512"))
    TRANSLATION_USE_SAFETENSORS_ONLY: bool = os.getenv("TRANSLATION_USE_SAFETENSORS_ONLY", "true").lower() in {"1",
                                                                                                               "true",
                                                                                                               "yes"}
    SUMMARY_S3_MODEL_URI: str = os.getenv("SUMMARY_S3_MODEL_URI", "")
    SUMMARY_MODEL_TMP_DIR: str = os.getenv("SUMMARY_MODEL_TMP_DIR", "/tmp/sa_models/summary")
    REQUIRE_S3: bool = os.getenv("REQUIRE_S3", "0").lower() in {"1", "true", "yes"}

    SUMMARY_CTX_SAFETY_RATIO: float = float(os.getenv("SUMMARY_CTX_SAFETY_RATIO", "0.8"))
    SUMMARY_TOPK_ENABLE: bool = os.getenv("SUMMARY_TOPK_ENABLE", "true").lower() in {"1", "true", "yes"}
    SUMMARY_TOPK_K: int = int(os.getenv("SUMMARY_TOPK_K", "1"))
    # ──────────────────────────────────────────────────────────────────
    #                       QUESTIONS (QG) — PT FLAN
    # ──────────────────────────────────────────────────────────────────
    # Modèle principal et fallback HF (utilisés par pipelines_questions*.py)
    PT_QG_MODEL: str = os.getenv("PT_QG_MODEL", "google/flan-t5-large")
    PT_QG_HF_MODEL: str = os.getenv("PT_QG_HF_MODEL", "google/flan-t5-large")

    # Chunking pour QG
    QG_PT_MAX_TOKENS: int = int(os.getenv("QG_PT_MAX_TOKENS", "800"))
    QG_PT_OVERLAP: int    = int(os.getenv("QG_PT_OVERLAP", "120"))

    # Tronquage dur en entrée QG
    QG_MAX_CHARS: int = int(os.getenv("QG_MAX_CHARS", "20000"))

    # Filtre QA
    ENABLE_QA_FILTER: bool = os.getenv("ENABLE_QA_FILTER", "false").lower() in {"1", "true", "yes"}


cfg = Config()
