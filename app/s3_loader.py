# app/s3_loader.py
from __future__ import annotations
import os, json, pathlib, boto3

DEFAULT_BUCKET    = os.getenv("S3_MODELS_BUCKET", os.getenv("S3_BUCKET", "smart-assistant-bucket"))
MODELS_MANIFEST   = os.getenv("MODELS_MANIFEST_JSON")   # fichier ou JSON inline
PREFIXES_MANIFEST = os.getenv("MODELS_PREFIXES_JSON")   # fichier ou JSON inline
FORCE_DOWNLOAD    = os.getenv("FORCE_DOWNLOAD", "0").lower() in {"1","true","yes"}
AWS_REGION        = os.getenv("AWS_REGION", "eu-west-3")

def _load_json(val: str|None):
    if not val: return None
    p = pathlib.Path(val)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(val)

def _s3():
    return boto3.client("s3", region_name=AWS_REGION)

def _ensure_dir(p: str|pathlib.Path):
    p = pathlib.Path(p); p.mkdir(parents=True, exist_ok=True); return p

def download_from_s3(bucket: str, key: str, local_path: str):
    dst = pathlib.Path(local_path)
    if dst.exists() and not FORCE_DOWNLOAD:
        return str(dst)
    _ensure_dir(dst.parent)
    _s3().download_file(bucket, key, str(dst))
    return str(dst)

def download_prefix(bucket: str, prefix: str, dest_dir: str):
    dest = _ensure_dir(dest_dir)
    paginator = _s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"): 
                continue
            rel = key[len(prefix):].lstrip("/")
            dst = dest / rel
            _ensure_dir(dst.parent)
            _s3().download_file(bucket, key, str(dst))
    return str(dest)

def download_all_models():
    bucket = DEFAULT_BUCKET

    manifest = _load_json(MODELS_MANIFEST)
    if isinstance(manifest, dict) and "files" in manifest:
        for f in manifest["files"]:
            download_from_s3(bucket, f["s3_key"], f["local_path"])

    prefixes_cfg = _load_json(PREFIXES_MANIFEST)
    if isinstance(prefixes_cfg, dict) and "prefixes" in prefixes_cfg:
        for entry in prefixes_cfg["prefixes"]:
            download_prefix(bucket, entry["prefix"], entry["dest_dir"])
