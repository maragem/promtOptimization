"""Web UI + API for the prompt-optimisation demo.

FastAPI backend serving:
- an eUI/EC-styled single-page interface (app/static)
- a small JSON API around the Haystack RAG pipeline and the LLM judges

Run locally:   uvicorn app.main:app --reload
On Railway:    see Dockerfile / railway.toml (uses the $PORT env var)
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config

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
    """Build the Chroma index on first boot (Railway's filesystem is ephemeral)."""
    from rag.retrieval import document_store

    store = document_store()
    if store.count_documents() == 0:
        logger.info("Chroma collection empty — indexing knowledge base...")
        from indexing.build_index import main as build_index

        build_index()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_index()
    except Exception:  # index lazily on first request instead of failing boot
        logger.exception("Indexing at startup failed; will retry on first request")
    yield


app = FastAPI(title="Prompt Optimisation Demo", lifespan=lifespan)


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
        "prompts": {
            "baseline": config.BASELINE_PROMPT_PATH.exists(),
            "optimized": config.OPTIMIZED_PROMPT_PATH.exists(),
        },
        "sample_questions": json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8")),
    }


@app.get("/api/prompt/{variant}")
def get_prompt(variant: str):
    return {"variant": variant, "text": load_prompt(variant)}


@app.post("/api/ask")
def ask(req: AskRequest):
    ensure_index()
    system_prompt = load_prompt(req.prompt)

    from rag.pipeline import answer

    try:
        reply, documents = answer(get_pipeline(), req.question, system_prompt)
    except Exception as exc:
        logger.exception("Pipeline call failed")
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}") from exc

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
            payload["scores"] = judge_answer(req.question, format_context(documents), reply)
        except Exception as exc:
            logger.exception("Judge call failed")
            payload["scores"] = {"error": f"Judge call failed: {exc}"}

    return payload


# Static frontend — mounted last so /api/* wins.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
