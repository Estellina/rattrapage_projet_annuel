# app/main.py
from __future__ import annotations

import logging, os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form, HTTPException, Request
from typing import Dict, Any
from fastapi import UploadFile, File, Form, HTTPException, Request
from typing import Dict, Any, List, Optional
import json

# Pipelines Questions
from app.pipelines.pipeline_questions import (
    run_questions_pipeline,
    run_questions_feedback_pipeline,
)



from app.config import cfg
from app import utils

# ──────────────────────────────────────────────────────────────────────────────
# Logs moins verbeux
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Chemins
APP_DIR       = Path(__file__).resolve().parent
STATIC_DIR    = APP_DIR / "static"        # <-- CSS/JS/Images
TEMPLATES_DIR = APP_DIR / "templates"     # <-- index.html / summary.html / questions.html

# ──────────────────────────────────────────────────────────────────────────────
# Pipelines "simples"
from app.pipelines.pipeline_summary import run_summary_pipeline

# Pipelines "feedback"
try:
    from app.pipelines.pipeline_summary_feedback import run_summary_feedback_pipeline
except Exception:
    run_summary_feedback_pipeline = None

# (optionnel) autres features questions/quiz
try:
    from app.pipelines.pipeline_questions import run_questions_pipeline
except Exception:
    run_questions_pipeline = None

try:
    from app.pipelines.pipeline_questions_feedback import run_questions_feedback_pipeline
except Exception:
    run_questions_feedback_pipeline = None

# Préwarm S3 (facultatif)
try:
    from app.s3_loader import download_all_models
except Exception:
    def download_all_models():
        return

# ──────────────────────────────────────────────────────────────────────────────
# Root logger
root = logging.getLogger()
if not root.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(h)
root.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    if cfg.PREWARM_MODELS:
        try:
            download_all_models()
            logging.info("[PREWARM] Modèles téléchargés en cache.")
        except Exception as e:
            logging.warning(f"[PREWARM] Ignoré: {e}")
    yield

# ──────────────────────────────────────────────────────────────────────────────
# App + middlewares
app = FastAPI(title=cfg.APP_NAME, lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=cfg.APP_SECRET)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cfg.DEBUG else [cfg.APP_BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers communs

import uuid
import threading

_PDF_CACHE = {}
_PDF_CACHE_LOCK = threading.RLock()
_PDF_CACHE_LIMIT = 16  # garde au plus 16 PDF (éviction FIFO simple)

def pdf_cache_put(pdf_bytes: bytes) -> str:
    doc_id = str(uuid.uuid4())
    with _PDF_CACHE_LOCK:
        if len(_PDF_CACHE) >= _PDF_CACHE_LIMIT:
            # éviction simple du plus ancien
            _PDF_CACHE.pop(next(iter(_PDF_CACHE)))
        _PDF_CACHE[doc_id] = pdf_bytes
    return doc_id

def pdf_cache_get(doc_id: str) -> bytes | None:
    with _PDF_CACHE_LOCK:
        return _PDF_CACHE.get(doc_id)


def _validate_pdf_upload(file: UploadFile, pdf_bytes: bytes) -> None:
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="PDF vide ou illisible.")
    max_bytes = cfg.MAX_UPLOAD_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (> {cfg.MAX_UPLOAD_MB} MB).")

def _preview_text_from_pdf(pdf_bytes: bytes) -> str:
    """Vérifie qu'on a bien du texte exploitable (pas un scan pur)."""
    preview = utils.extract_text_safe(pdf_bytes)
    if not preview:
        preview = utils.extract_text_from_pdf(pdf_bytes)
    return preview or ""

def _norm_summary(out: Any) -> Dict[str, Any]:
    if isinstance(out, dict):
        return {
            "summary_fr": out.get("text_fr") or out.get("summary_fr") or "",
            "summary_en": out.get("text_en") or out.get("summary_en") or "",
            "model_used": out.get("model_used", "unknown"),
            "quality": out.get("quality"),
        }
    return {
        "summary_fr": getattr(out, "text_fr", "") or getattr(out, "summary_fr", ""),
        "summary_en": getattr(out, "text_en", "") or getattr(out, "summary_en", ""),
        "model_used": getattr(out, "model_used", "unknown"),
        "quality": getattr(out, "quality", None),
    }

def _read_tpl(name: str) -> str:
    p = TEMPLATES_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Template manquant: {name}")
    return p.read_text(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# UI (templates)

@app.get("/", response_class=HTMLResponse)
def page_home() -> str:
    return _read_tpl("index.html")

@app.get("/summary", response_class=HTMLResponse)
def page_summary() -> str:
    return _read_tpl("summary.html")

@app.get("/questions", response_class=HTMLResponse)
def page_questions() -> str:
    return _read_tpl("questions.html")

# Favicon (si présent dans /static)
@app.get("/favicon.ico")
def favicon():
    ico = STATIC_DIR / "favicon.ico"
    if ico.exists():
        return Response(content=ico.read_bytes(), media_type="image/x-icon")
    return Response(status_code=404)

# Health
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)

# ──────────────────────────────────────────────────────────────────────────────
# API: Résumé PDF

from fastapi import Request
import tempfile, uuid, json
from pathlib import Path

from fastapi import Request

@app.post("/api/summary")
async def api_summary(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # 1) Lire le PDF uploadé
        pdf_bytes = await file.read()
        _validate_pdf_upload(file, pdf_bytes)

        # 2) Cacher le PDF en mémoire et obtenir doc_id
        doc_id = pdf_cache_put(pdf_bytes)

        # 3) Lancer la pipeline de résumé habituelle
        out = run_summary_pipeline(pdf_bytes)

        # 4) Normaliser la réponse côté client
        retval = _norm_summary(out)
        retval["ok"] = True
        retval["doc_id"] = doc_id  # ← important pour le feedback (frontend)

        # 5) (Optionnel) garder un cache disque pour fallback UI
        try:
            cache_dir = Path(tempfile.gettempdir()) / "sa_webcache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            p = cache_dir / f"{uuid.uuid4().hex}.pdf"
            p.write_bytes(pdf_bytes)
            request.session["pdf_cache_path"] = str(p)
        except Exception:
            logging.exception("failed to cache pdf")

        # 6) (Optionnel) conserver le dernier résumé pour préremplir le textarea
        try:
            request.session["summary_last"] = {
                "summary_fr": retval.get("summary_fr", ""),
                "summary_en": retval.get("summary_en", ""),
                "model_used": retval.get("model_used", ""),
                "route": "summary",
            }
        except Exception:
            pass

        return retval

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("summary failed")
        raise HTTPException(status_code=500, detail=f"summary failed: {e}")



# ──────────────────────────────────────────────────────────────────────────────
# API: Résumé (feedback) — lit les tokens UI + le texte FR courant (`source_fr`)



@app.post("/api/summary/feedback")
async def api_summary_feedback(
    request: Request,
    doc_id: str = Form(""),
    length: str = Form("default"),
    ton: str = Form("default"),
    focus: str = Form("default"),
    structure: str = Form("default"),
    couverture: str = Form("default"),
    style: str = Form("default"),
    chiffres: str = Form("default"),
    citations: str = Form("default"),
    source_fr: str = Form(""),
    topk_k: str = Form("2"),
) -> Dict[str, Any]:
    try:
        # 0) Récupérer le PDF depuis le cache mémoire via doc_id
        if not doc_id:
            raise HTTPException(status_code=400, detail="doc_id manquant : relancez la première étape de résumé.")
        pdf_bytes = pdf_cache_get(doc_id)
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Aucun PDF en cache pour ce doc_id.")

        # 1) Préférences 'sparse'
        prefs: Dict[str, str] = {}
        if length not in ("", "default"):     prefs["length"] = length
        if ton not in ("", "default"):        prefs["ton"] = ton
        if focus not in ("", "default"):      prefs["focus"] = focus
        if structure not in ("", "default"):  prefs["structure"] = structure
        if couverture not in ("", "default"): prefs["couverture"] = couverture
        if style not in ("", "default"):      prefs["style"] = style
        if chiffres not in ("", "default"):   prefs["chiffres"] = chiffres
        if citations not in ("", "default"):  prefs["citations"] = citations
        if source_fr.strip():                  prefs["source_fr"] = source_fr.strip()
        if topk_k.strip():                     prefs["topk_k"] = topk_k.strip()

        # 2) Lancer la nouvelle pipeline de feedback (qui lit le PDF)
        out = run_summary_feedback_pipeline(pdf_bytes, prefs)

        # 3) Réponse UI (format identique au résumé)
        return _norm_summary(out)

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("summary feedback failed")
        raise HTTPException(status_code=500, detail=f"summary feedback failed: {e}")



# ──────────────────────────────────────────────────────────────────────────────
# (Optionnel) API: Questions
@app.post("/api/questions")
async def api_questions(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # 1) lire & valider le PDF
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="PDF vide ou illisible.")

        # 2) cache mémoire → doc_id (même logique que pour /api/summary)
        doc_id = pdf_cache_put(pdf_bytes)

        # 3) lancer la pipeline de questions (génération standard)
        default_n = int(os.getenv("QG_DEFAULT_NUM_QUESTIONS", "5"))
        out = run_questions_pipeline(pdf_bytes, num_questions=default_n, doc_id=doc_id)

        # 4) réponse UI
        result = {
            "ok": True,
            "questions_fr": out.questions_fr,
            "questions_en": out.questions_en,
            "model_used": out.model_used,
            "doc_id": doc_id
        }

        # (optionnel) garder pour debug/UX
        try:
            request.session["qg_last"] = result
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("questions failed")
        raise HTTPException(status_code=500, detail=f"questions failed: {e}")


@app.post("/api/questions/feedback")
async def api_questions_feedback(
    request: Request,
    doc_id: str = Form(""),
    payload: str = Form("{}"),
) -> Dict[str, Any]:
    try:
        # 1) payload JSON -> dict
        try:
            pld = json.loads(payload) if payload else {}
        except Exception:
            pld = {}
        logging.info("[QG][API] payload=%s", payload)

        # 2) doc_id : accepter celui du formulaire OU celui dans le payload
        if not doc_id:
            doc_id = pld.get("doc_id", "") or ""

        if not doc_id:
            raise HTTPException(status_code=400, detail="doc_id manquant (relancez la génération initiale).")

        # 3) retrouver le PDF depuis le cache mémoire
        pdf_bytes = pdf_cache_get(doc_id)
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Aucun PDF en cache pour ce doc_id (relancez la génération initiale).")

        # 4) lancer la pipeline de feedback (reasons → règles, spans figées par doc_id côté pipeline)
        out = run_questions_feedback_pipeline(
            pdf_bytes=pdf_bytes,
            payload=pld,
        )

        # 5) réponse UI
        return {
            "ok": True,
            "questions_fr": out.questions_fr,
            "questions_en": out.questions_en,
            "model_used": out.model_used,
            "doc_id": doc_id
        }



    except HTTPException:
        raise
    except Exception as e:
        logging.exception("questions feedback failed")
        raise HTTPException(status_code=500, detail=f"questions feedback failed: {e}")



from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Run local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=cfg.DEBUG)
