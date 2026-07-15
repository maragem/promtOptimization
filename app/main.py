"""Web UI + API for the prompt-optimisation demo.

FastAPI backend serving:
- an eUI/EC-styled single-page interface (app/static)
- a small JSON API around the Haystack RAG pipeline and the LLM judges

Run locally:   uvicorn app.main:app --reload
On Railway:    see Dockerfile / railway.toml (uses the $PORT env var)
"""

import hashlib
import hmac
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag.pipeline import build_pipeline

        _pipeline = build_pipeline()
    return _pipeline


def ensure_index() -> None:
    """Build the Chroma index if missing (Railway's filesystem is ephemeral)."""
    from rag.retrieval import document_store

    store = document_store()
    if store.count_documents() == 0:
        logger.info("Chroma collection empty — indexing knowledge base...")
        from indexing.build_index import main as build_index

        build_index()


# Nothing is loaded at startup: the embedding model, the index, and the
# Bedrock connection are all initialised on demand via POST /api/init.
app = FastAPI(title="Prompt Optimisation Demo")

_ready = False

# --- Access gate --------------------------------------------------------------
# Enabled by setting APP_PASSWORD. Browser sessions authenticate via the login
# page (HttpOnly cookie); scripts can send "Authorization: Bearer <password>".

SESSION_COOKIE = "demo_session"
SESSION_MAX_AGE = 12 * 3600
# Paths that stay reachable without auth (healthcheck, login flow, page assets).
PUBLIC_PATHS = {"/api/health", "/api/login", "/login.html", "/styles.css"}


def session_token() -> str:
    return hmac.new(
        config.APP_PASSWORD.encode(), b"prompt-optimisation-demo-session", hashlib.sha256
    ).hexdigest()


def is_authenticated(request: Request) -> bool:
    cookie = request.cookies.get(SESSION_COOKIE, "")
    if cookie and hmac.compare_digest(cookie, session_token()):
        return True
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return hmac.compare_digest(auth[7:], config.APP_PASSWORD)
    return False


@app.middleware("http")
async def access_gate(request: Request, call_next):
    if config.APP_PASSWORD and request.url.path not in PUBLIC_PATHS:
        if not is_authenticated(request):
            if request.url.path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
            return RedirectResponse(url="/login.html", status_code=303)
    return await call_next(request)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=200)


@app.post("/api/login")
def login(req: LoginRequest):
    if not config.APP_PASSWORD:
        return {"status": "open"}  # no gate configured
    if not hmac.compare_digest(req.password, config.APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect password")
    response = JSONResponse(content={"status": "ok"})
    response.set_cookie(
        SESSION_COOKIE,
        session_token(),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse(content={"status": "ok"})
    response.delete_cookie(SESSION_COOKIE)
    return response



class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    prompt: str = "baseline"  # "baseline" | "optimized"
    judge: bool = False


def load_prompt(variant: str) -> str:
    path = {
        "baseline": config.BASELINE_PROMPT_PATH,
        "optimized": config.OPTIMIZED_PROMPT_PATH,
    }.get(variant)
    if path is None or not path.exists():
        raise HTTPException(status_code=400, detail=f"Unknown or missing prompt variant: {variant}")
    return path.read_text(encoding="utf-8").strip()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return {
        "production_model": config.PROD_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "region": config.AWS_REGION,
        "dataset": config.DATASET_ID,
        "dataset_label": config.DATASETS[config.DATASET_ID]["label"],
        "prompts": {
            "baseline": config.BASELINE_PROMPT_PATH.exists(),
            "optimized": config.OPTIMIZED_PROMPT_PATH.exists(),
        },
        "sample_questions": json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8")),
    }


@app.get("/api/prompt/{variant}")
def get_prompt(variant: str):
    return {"variant": variant, "text": load_prompt(variant)}


@app.get("/api/status")
def status():
    return {"ready": _ready, "model": config.PROD_MODEL, "region": config.AWS_REGION}


@app.post("/api/init")
def init_assistant():
    """Manual initialisation, triggered from the UI after login.

    Stage 1: download the embedding model (first run) and build the index.
    Stage 2: build the pipeline and make one minimal Bedrock test call.
    """
    global _ready
    start = time.perf_counter()

    try:
        ensure_index()
    except Exception as exc:
        logger.exception("Initialisation failed at the indexing stage")
        return JSONResponse(
            status_code=502,
            content={"status": "error", "stage": "index", "error": f"{type(exc).__name__}: {exc}"},
        )
    indexed_at = time.perf_counter()

    from haystack.dataclasses import ChatMessage

    try:
        llm = get_pipeline().get_component("llm")
        result = llm.run(messages=[ChatMessage.from_user("Reply with the single word: pong")])
        reply = result["replies"][0].text
    except Exception as exc:
        logger.exception("Initialisation failed at the model stage")
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "stage": "model",
                "model": config.PROD_MODEL,
                "region": config.AWS_REGION,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    _ready = True
    logger.info("Initialised: index %.1fs, model test %.1fs", indexed_at - start, time.perf_counter() - indexed_at)
    return {
        "status": "ok",
        "model": config.PROD_MODEL,
        "region": config.AWS_REGION,
        "index_s": round(indexed_at - start, 1),
        "model_s": round(time.perf_counter() - indexed_at, 1),
        "reply": reply,
    }


# --- Datasets ------------------------------------------------------------------


def _dataset_entries() -> list[dict]:
    entries = []
    for ds_id, info in config.DATASETS.items():
        questions_path = info["dir"] / "questions.json"
        available = (info["dir"] / "knowledge_base.json").exists() and questions_path.exists()
        n_questions, golden = None, False
        if available:
            questions = json.loads(questions_path.read_text(encoding="utf-8"))
            n_questions = len(questions)
            golden = any(q.get("answer") for q in questions)
        entries.append(
            {
                "id": ds_id,
                "label": info["label"],
                "available": available,
                "questions": n_questions,
                "golden": golden,
                "active": ds_id == config.DATASET_ID,
            }
        )
    return entries


@app.get("/api/datasets")
def list_datasets():
    return {"active": config.DATASET_ID, "datasets": _dataset_entries()}


class DatasetRequest(BaseModel):
    id: str


@app.post("/api/dataset")
def switch_dataset(req: DatasetRequest):
    """Switch the active dataset; fetches the golden set on first use."""
    global _pipeline
    if req.id not in config.DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {req.id}")
    with _opt_lock:
        if _opt_job.get("status") == "running":
            raise HTTPException(status_code=409, detail="Cannot switch datasets while an optimisation is running")

    target_dir = config.DATASETS[req.id]["dir"]
    if not (target_dir / "knowledge_base.json").exists():
        if req.id != "golden":
            raise HTTPException(status_code=400, detail=f"Dataset files missing for '{req.id}'")
        logger.info("Golden dataset not on disk — fetching from Hugging Face...")
        from indexing.fetch_hf_golden import fetch_golden

        try:
            fetch_golden(target_dir)
        except Exception as exc:
            logger.exception("Golden dataset fetch failed")
            raise HTTPException(status_code=502, detail=f"Dataset fetch failed: {type(exc).__name__}: {exc}") from exc

    previous = config.DATASET_ID
    config.set_dataset(req.id)
    _pipeline = None  # rebuild against the new Chroma collection
    try:
        ensure_index()
    except Exception as exc:
        config.set_dataset(previous)
        _pipeline = None
        logger.exception("Indexing the new dataset failed; reverted to %s", previous)
        raise HTTPException(status_code=502, detail=f"Indexing failed: {type(exc).__name__}: {exc}") from exc

    logger.info("Active dataset switched to %s", req.id)
    return {"status": "ok", "active": config.DATASET_ID, "datasets": _dataset_entries()}


def gold_answer_for(question: str) -> str | None:
    """Look up the gold reference answer when the asked question is from the dataset."""
    try:
        for q in json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8")):
            if q.get("answer") and q["question"].strip().lower() == question.strip().lower():
                return q["answer"]
    except FileNotFoundError:
        pass
    return None


# --- Prompt optimisation as a background job ----------------------------------
# GEPA takes many minutes and hundreds of model calls, so it runs in a
# daemon thread; the UI polls /api/optimize/status for live progress.

_opt_lock = threading.Lock()
_opt_job = {"status": "idle"}


def _fresh_job(budget: str) -> dict:
    return {
        "status": "running",
        "budget": budget,
        "dataset": config.DATASET_ID,
        "started_at": time.time(),
        "metric_calls": 0,
        "last_scores": None,
        "error": None,
        "prompt": None,
    }


def _run_optimisation_job(budget: str) -> None:
    try:
        ensure_index()
        from optimisation.run_gepa import run_optimisation

        def progress(result: dict) -> None:
            with _opt_lock:
                _opt_job["metric_calls"] += 1
                _opt_job["last_scores"] = {
                    k: v for k, v in result.items() if k not in ("feedback",)
                }

        prompt = run_optimisation(budget=budget, progress=progress)
        with _opt_lock:
            _opt_job.update(status="done", prompt=prompt, finished_at=time.time())
        logger.info("Optimisation finished after %d metric calls", _opt_job["metric_calls"])
    except Exception as exc:
        logger.exception("Optimisation job failed")
        with _opt_lock:
            _opt_job.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time(),
            )


class OptimizeRequest(BaseModel):
    budget: str = Field(default="light", pattern="^(light|medium|heavy)$")


@app.post("/api/optimize")
def start_optimization(req: OptimizeRequest):
    global _opt_job
    with _opt_lock:
        if _opt_job.get("status") == "running":
            raise HTTPException(status_code=409, detail="An optimisation run is already in progress")
        _opt_job = _fresh_job(req.budget)
    threading.Thread(target=_run_optimisation_job, args=(req.budget,), daemon=True).start()
    return {"status": "started", "budget": req.budget}


@app.get("/api/optimize/status")
def optimization_status():
    with _opt_lock:
        job = dict(_opt_job)
    if job.get("started_at"):
        job["elapsed_s"] = round((job.get("finished_at") or time.time()) - job["started_at"], 1)
    return job


@app.get("/api/test-model")
def test_model():
    """Diagnostic: one tiny Bedrock call, returns latency or the real error."""
    from haystack.dataclasses import ChatMessage

    pipe = get_pipeline()
    llm = pipe.get_component("llm")
    start = time.perf_counter()
    try:
        result = llm.run(messages=[ChatMessage.from_user("Reply with the single word: pong")])
        return {
            "status": "ok",
            "model": config.PROD_MODEL,
            "region": config.AWS_REGION,
            "latency_s": round(time.perf_counter() - start, 2),
            "reply": result["replies"][0].text,
        }
    except Exception as exc:
        logger.exception("Model connectivity test failed")
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "model": config.PROD_MODEL,
                "region": config.AWS_REGION,
                "latency_s": round(time.perf_counter() - start, 2),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


@app.post("/api/ask")
def ask(req: AskRequest):
    ensure_index()
    system_prompt = load_prompt(req.prompt)

    from rag.pipeline import answer

    logger.info("ask: start (prompt=%s, model=%s, region=%s)", req.prompt, config.PROD_MODEL, config.AWS_REGION)
    start = time.perf_counter()
    try:
        reply, documents = answer(get_pipeline(), req.question, system_prompt)
    except Exception as exc:
        logger.exception("Pipeline call failed after %.1fs", time.perf_counter() - start)
        raise HTTPException(status_code=502, detail=f"Model call failed: {type(exc).__name__}: {exc}") from exc
    logger.info("ask: model replied in %.1fs", time.perf_counter() - start)

    payload = {
        "answer": reply,
        "prompt": req.prompt,
        "model": config.PROD_MODEL,
        "documents": [
            {"id": d.id, "content": d.content, "score": getattr(d, "score", None)}
            for d in documents
        ],
        "scores": None,
    }

    if req.judge:
        from optimisation.metrics import judge_answer
        from rag.retrieval import format_context

        try:
            payload["scores"] = judge_answer(
                req.question, format_context(documents), reply,
                gold_answer=gold_answer_for(req.question),
            )
        except Exception as exc:
            logger.exception("Judge call failed")
            payload["scores"] = {"error": f"Judge call failed: {exc}"}

    return payload


# Static frontend — mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
