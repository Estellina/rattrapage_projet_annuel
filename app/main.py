from __future__ import annotations

"""
app/main.py — version nettoyée et structurée
- Imports dédupliqués et triés
- Sections claires (config, cache PDF, helpers, routes UI, API)
- Comportement inchangé
"""

# ============================================================================
# Imports standard
# ============================================================================
import json
import logging
import os
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# Imports FastAPI / Starlette
# ============================================================================
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware

# ============================================================================
# Imports applicatifs
# ============================================================================
from app import utils
from app.config import cfg
from app.pipelines.pipeline_summary import run_summary_pipeline

# Pipelines optionnelles (feedback / questions)
try:
    from app.pipelines.pipeline_summary_feedback import run_summary_feedback_pipeline  # type: ignore
except Exception:  # pragma: no cover - pipeline optionnelle
    run_summary_feedback_pipeline = None  # type: ignore

try:
    from app.pipelines.pipeline_questions import run_questions_pipeline  # type: ignore
except Exception:  # pragma: no cover - pipeline optionnelle
    run_questions_pipeline = None  # type: ignore

try:
    from app.pipelines.pipeline_questions_feedback import run_questions_feedback_pipeline  # type: ignore
except Exception:  # pragma: no cover - pipeline optionnelle
    run_questions_feedback_pipeline = None  # type: ignore

# Préwarm S3 optionnel
try:
    from app.s3_loader import download_all_models  # type: ignore
except Exception:  # pragma: no cover - s3 optionnel
    def download_all_models() -> None:  # type: ignore
        return

# ============================================================================
# Logging
# ============================================================================
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

root = logging.getLogger()
if not root.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(handler)
root.setLevel(logging.DEBUG if cfg.DEBUG else logging.INFO)

# ============================================================================
# Dossiers applicatifs
# ============================================================================
APP_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = APP_DIR / "static"
TEMPLATES_DIR: Path = APP_DIR / "templates"

# ============================================================================
# Lifespan (pré-chargement des modèles)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if cfg.PREWARM_MODELS:
        try:
            download_all_models()
            logging.info("[PREWARM] Modèles téléchargés en cache.")
        except Exception as e:  # pragma: no cover
            logging.warning(f"[PREWARM] Ignoré: {e}")
    yield

# ============================================================================
# App + middlewares
# ============================================================================
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

# ============================================================================
# Cache PDF en mémoire (FIFO simple)
# ============================================================================
_PDF_CACHE: Dict[str, bytes] = {}
_PDF_CACHE_LOCK = threading.RLock()
_PDF_CACHE_LIMIT = 16  # conserve au plus 16 PDF


def pdf_cache_put(pdf_bytes: bytes) -> str:
    """Ajoute le PDF au cache et renvoie un doc_id unique."""
    doc_id = uuid.uuid4().hex
    with _PDF_CACHE_LOCK:
        if len(_PDF_CACHE) >= _PDF_CACHE_LIMIT:
            # éviction du plus ancien (ordre d'itération des dicts = insertion)
            _PDF_CACHE.pop(next(iter(_PDF_CACHE)))
        _PDF_CACHE[doc_id] = pdf_bytes
    return doc_id


def pdf_cache_get(doc_id: str) -> Optional[bytes]:
    """Récupère le PDF depuis le cache mémoire, sinon None."""
    with _PDF_CACHE_LOCK:
        return _PDF_CACHE.get(doc_id)

# ============================================================================
# Helpers
# ============================================================================

def _validate_pdf_upload(file: UploadFile, pdf_bytes: bytes) -> None:
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="PDF vide ou illisible.")
    max_bytes = cfg.MAX_UPLOAD_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (> {cfg.MAX_UPLOAD_MB} MB).")


def _preview_text_from_pdf(pdf_bytes: bytes) -> str:
    """Retourne un texte d'aperçu (favorise pypdf, fallback pdfminer)."""
    preview = utils.extract_text_safe(pdf_bytes)
    if not preview:
        preview = utils.extract_text_from_pdf(pdf_bytes)
    return preview or ""


def _norm_summary(out: Any) -> Dict[str, Any]:
    """Normalise la sortie de pipeline en un dict stable pour l'UI."""
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

# ============================================================================
# Routes UI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def page_home() -> str:
    return _read_tpl("index.html")


@app.get("/summary", response_class=HTMLResponse)
def page_summary() -> str:
    return _read_tpl("summary.html")


@app.get("/questions", response_class=HTMLResponse)
def page_questions() -> str:
    return _read_tpl("questions.html")


# Favicon
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

# ============================================================================
# API Résumé
# ============================================================================

@app.post("/api/summary")
async def api_summary(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        # 1) Lire le PDF uploadé
        pdf_bytes = await file.read()
        _validate_pdf_upload(file, pdf_bytes)

        # 2) Cache mémoire → doc_id
        doc_id = pdf_cache_put(pdf_bytes)

        # 3) Lancer la pipeline de résumé
        out = run_summary_pipeline(pdf_bytes)

        # 4) Normaliser la réponse côté client
        retval: Dict[str, Any] = _norm_summary(out)
        retval["ok"] = True
        retval["doc_id"] = doc_id  # pour la route feedback

        # 5) (Optionnel) cache disque pour fallback UI
        try:
            cache_dir = Path(tempfile.gettempdir()) / "sa_webcache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            p = cache_dir / f"{uuid.uuid4().hex}.pdf"
            p.write_bytes(pdf_bytes)
            request.session["pdf_cache_path"] = str(p)
        except Exception:  # pragma: no cover - best effort
            logging.exception("failed to cache pdf")

        # 6) (Optionnel) conserver le dernier résumé pour pré-remplir le textarea
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
        if not doc_id:
            raise HTTPException(status_code=400, detail="doc_id manquant : relancez la première étape de résumé.")
        pdf_bytes = pdf_cache_get(doc_id)
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Aucun PDF en cache pour ce doc_id.")

        # Préférences 'sparse'
        prefs: Dict[str, str] = {}
        if length not in ("", "default"):     prefs["length"] = length
        if ton not in ("", "default"):        prefs["ton"] = ton
        if focus not in ("", "default"):      prefs["focus"] = focus
        if structure not in ("", "default"):  prefs["structure"] = structure
        if couverture not in ("", "default"): prefs["couverture"] = couverture
        if style not in ("", "default"):      prefs["style"] = style
        if chiffres not in ("", "default"):   prefs["chiffres"] = chiffres
        if citations not in ("", "default"):  prefs["citations"] = citations
        if source_fr.strip():                    prefs["source_fr"] = source_fr.strip()
        if topk_k.strip():                       prefs["topk_k"] = topk_k.strip()

        if run_summary_feedback_pipeline is None:
            raise HTTPException(status_code=501, detail="Route feedback indisponible sur ce déploiement.")

        out = run_summary_feedback_pipeline(pdf_bytes, prefs)
        return _norm_summary(out)

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("summary feedback failed")
        raise HTTPException(status_code=500, detail=f"summary feedback failed: {e}")


# ============================================================================
# API Questions
# ============================================================================

@app.post("/api/questions")
async def api_questions(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="PDF vide ou illisible.")

        doc_id = pdf_cache_put(pdf_bytes)

        if run_questions_pipeline is None:
            raise HTTPException(status_code=501, detail="Génération de questions indisponible sur ce déploiement.")

        default_n = int(os.getenv("QG_DEFAULT_NUM_QUESTIONS", "5"))
        out = run_questions_pipeline(pdf_bytes, num_questions=default_n, doc_id=doc_id)

        result = {
            "ok": True,
            "questions_fr": out.questions_fr,
            "questions_en": out.questions_en,
            "model_used": out.model_used,
            "doc_id": doc_id,
        }

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

        if run_questions_feedback_pipeline is None:
            raise HTTPException(status_code=501, detail="Feedback questions indisponible sur ce déploiement.")

        # 4) lancer la pipeline de feedback
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
            "doc_id": doc_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("questions feedback failed")
        raise HTTPException(status_code=500, detail=f"questions feedback failed: {e}")


# ============================================================================
# Entrée locale
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=cfg.DEBUG)
