from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
from .pipelines.pipeline_summary import run_summary_pipeline
from .pipelines.pipeline_questions import run_questions_pipeline
from .pipelines.pipeline_summary_feedback import run_summary_feedback_pipeline
from .pipelines.pipeline_questions_feedback import run_questions_feedback_pipeline
import json
from pathlib import Path

app = FastAPI(title="Smart Assistant")

# Chemins robustes (fonctionne local + EB)
ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def _html(name: str) -> HTMLResponse:
    return HTMLResponse((TEMPLATES_DIR / name).read_text(encoding="utf-8"))

# --------- VUES (UI) ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return _html("index.html")


@app.get("/summary", response_class=HTMLResponse)
def summary_page():
    return _html("summary.html")

@app.get("/questions", response_class=HTMLResponse)
def questions_page():
    return _html("questions.html")

# --------- API PIPELINES ----------
@app.post("/api/summary")
async def api_summary(file: UploadFile):
    data = await file.read()
    out = run_summary_pipeline(data)
    return JSONResponse({
        "summary_fr": out.text_fr,
        "summary_en": out.text_en,
        "model_used": out.model_used,
        "quality": out.quality
    })

@app.post("/api/questions")
async def api_questions(file: UploadFile, n: int = 5):
    data = await file.read()
    out = run_questions_pipeline(data, num_questions=n)
    return JSONResponse({
        "questions_fr": out.questions_fr,
        "questions_en": out.questions_en,
        "model_used": out.model_used
    })

@app.post("/api/summary/feedback")
async def api_summary_feedback(
    source_fr: str = Form(...),
    length: str = Form("standard"),
    ton: str = Form("neutre"),
    focus: str = Form("général"),
    structure: str = Form("paragraphes"),
    couverture: str = Form("concis"),
    style: str = Form("abstractive"),
    chiffres: str = Form("garder"),
    citations: str = Form("inclure"),
):
    fb = dict(length=length, ton=ton, focus=focus, structure=structure,
              couverture=couverture, style=style, chiffres=chiffres, citations=citations)
    out_fr = run_summary_feedback_pipeline(source_fr, fb)
    return JSONResponse({"summary_fr": out_fr})

@app.post("/api/questions/feedback")
async def api_questions_feedback(
    source_fr: str = Form(...),
    labels: List[str] = Form([]),
    payload: str = Form("{}"),
    questions_fr: List[str] = Form([]),
):
    try:
        pl = json.loads(payload) if payload else {}
    except Exception:
        pl = {}
    out_fr = run_questions_feedback_pipeline(source_fr, questions_fr, labels, pl)
    return JSONResponse({"questions_fr": out_fr})
