# app/config.py
import os
from dataclasses import dataclass
from pathlib import Path

def _to_dir(uri: str | None) -> str:
    """Prend un URI S3 ou un chemin et renvoie son 'dossier' (avec / final)."""
    if not uri:
        return ""
    if uri.startswith("s3://"):
        return uri if uri.endswith("/") else uri.rsplit("/", 1)[0] + "/"
    return uri if uri.endswith("/") else uri + "/"

# Aligne le cache HF/Transformers pour éviter les PermissionError
_hf_cache = os.getenv("TRANSFORMERS_CACHE") or "/tmp/transformers_cache"
os.environ["TRANSFORMERS_CACHE"] = _hf_cache
os.environ["HF_HOME"] = _hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache
Path(_hf_cache).mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    # Commun
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://127.0.0.1:8080")
    AWS_REGION:  str = os.getenv("AWS_REGION", "eu-west-3")
    S3_BUCKET:   str = os.getenv("S3_BUCKET", "smart-assistant-bucket")
    S3_PREFIX:   str = os.getenv("S3_PREFIX", "")

    # --- Summary (from-scratch) : noms EXACTS EB ---
    FS_SUMMARY_CKPT: str = os.getenv("SUM_FS_CHECKPOINT_S3", "")
    FS_SUMMARY_TOK:  str = (
        _to_dir(os.getenv("SUM_FS_TOKENIZER_MERGES_S3"))
        or _to_dir(os.getenv("SUM_FS_TOKENIZER_VOCAB_S3"))
    )

    # --- QG (from-scratch) : noms EXACTS EB ---
    FS_QG_CKPT: str = os.getenv("QG_FS_CHECKPOINT_S3", "")
    FS_QG_TOK:  str = (
        _to_dir(os.getenv("QG_FS_TOKENIZER_MERGES_S3"))
        or _to_dir(os.getenv("QG_FS_TOKENIZER_VOCAB_S3"))
    )

    # --- Pré-entraîné ---
    PT_QG_MODEL: str = os.getenv("PT_QG_MODEL", "google/flan-t5-large")
    # EB donne PT_SUMMARY_S3_CHECKPOINT (fichier ou dossier) -> on fabrique un préfixe
    PT_SUMMARY_S3_PREFIX: str = _to_dir(os.getenv("PT_SUMMARY_S3_CHECKPOINT"))

    # Divers
    ENABLE_QA_FILTER: bool = os.getenv("TOKENIZERS_PARALLELISM", "false").lower() in {"1","true","yes"} \
                             or os.getenv("ENABLE_QA_FILTER", "false").lower() in {"1","true","yes"}
    ENABLE_OFFLINE_TRANSLATION: bool = os.getenv("ENABLE_OFFLINE_TRANSLATION", "false").lower() in {"1","true","yes"}
    TRANSLATION_SRC: str = os.getenv("TRANSLATION_SRC", "fr")
    TRANSLATION_TGT: str = os.getenv("TRANSLATION_TGT", "en")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/sa_cache")

cfg = Config()
