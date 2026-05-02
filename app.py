#!/usr/bin/env python3
"""
app.py  –  FastAPI backend + Jinja2 dashboard

Routes:
  GET  /              → dashboard (renders index.html with last eval results)
  POST /analyze       → live single-prompt detection (returns JSON)
  GET  /results       → raw evaluation_results.json as JSON API

Run:
  uvicorn app:app --reload --port 8000
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="Prompt Injection Detector")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

RESULTS_FILE = Path("evaluation_results.json")


# ── Lazy-load LangKit once at startup ─────────────────────────────────────────

_extract_fn = None

def get_extract():
    global _extract_fn
    if _extract_fn is None:
        try:
            from langkit import injections, jailbreaks
            from langkit.extract import extract
            _extract_fn = extract
        except ModuleNotFoundError:
            return None
    return _extract_fn


INJECTION_THRESHOLD = 0.50
JAILBREAK_THRESHOLD = 0.50


def _score(prompt: str) -> dict:
    extract = get_extract()
    if extract is None:
        return {"error": "LangKit not installed"}

    raw = extract({"prompt": prompt})

    def pick(keys):
        for k in keys:
            v = raw.get(k)
            if v is not None:
                return round(float(v), 4)
        return None

    inj = pick(["prompt.injection_similarity", "prompt.injection", "injection_similarity", "injection"])
    jb  = pick(["prompt.jailbreak_similarity",  "prompt.jailbreak",  "jailbreak_similarity",  "jailbreak"])

    flagged = (inj or 0) >= INJECTION_THRESHOLD or (jb or 0) >= JAILBREAK_THRESHOLD

    reasons = []
    if inj and inj >= INJECTION_THRESHOLD:
        reasons.append(f"Injection score {inj} ≥ {INJECTION_THRESHOLD}")
    if jb and jb >= JAILBREAK_THRESHOLD:
        reasons.append(f"Jailbreak score {jb} ≥ {JAILBREAK_THRESHOLD}")

    return {
        "input":            prompt,
        "injection_score":  inj,
        "jailbreak_score":  jb,
        "flagged":          flagged,
        "action":           "BLOCKED" if flagged else "ALLOWED",
        "reasons":          reasons,
    }


def _load_eval_results() -> dict | None:
    if not RESULTS_FILE.exists():
        return None
    try:
        return json.loads(RESULTS_FILE.read_text())
    except Exception:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    eval_data = _load_eval_results()
    metrics   = eval_data["metrics"] if eval_data else None
    records   = eval_data["records"][:20] if eval_data else []   # show latest 20 in table

    return templates.TemplateResponse("index.html", {
        "request": request,
        "metrics": metrics,
        "records": records,
        "has_results": eval_data is not None,
    })


@app.post("/analyze", response_class=JSONResponse)
async def analyze(prompt: str = Form(...)):
    result = _score(prompt)
    return JSONResponse(content=result)


@app.get("/results", response_class=JSONResponse)
async def results():
    data = _load_eval_results()
    if data is None:
        return JSONResponse(
            content={"error": "No evaluation results yet. Run evaluator.py first."},
            status_code=404,
        )
    return JSONResponse(content=data)
