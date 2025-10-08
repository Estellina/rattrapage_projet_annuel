
import pathlib, boto3
from urllib.parse import urlparse

def parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri}")
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")

def ensure_dir(p):
    p = pathlib.Path(p); p.mkdir(parents=True, exist_ok=True); return p

def download_s3_file(uri: str, local_path):
    bucket, key = parse_s3_uri(uri)
    local_path = pathlib.Path(local_path)
    ensure_dir(local_path.parent)
    boto3.client("s3").download_file(bucket, key, str(local_path))
    return str(local_path)

def download_s3_prefix(uri_prefix: str, local_dir):
    bucket, prefix = parse_s3_uri(uri_prefix)
    local_dir = ensure_dir(local_dir)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):].lstrip("/")
            dst = local_dir / rel
            ensure_dir(dst.parent)
            s3.download_file(bucket, key, str(dst))
    return str(local_dir)
