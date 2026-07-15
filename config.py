"""Central configuration: models, paths, and LM factories.

Everything provider-specific lives here so the later switch from
Claude-on-Bedrock to GPT@EC is a change to this one file (see README).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent

# --- AWS / Bedrock -----------------------------------------------------------
# Auth is a Bedrock API key (bearer token) in AWS_BEARER_TOKEN_BEDROCK.
# boto3 >= 1.39 and LiteLLM pick it up from the environment automatically.
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_REGION)

# In EU regions Bedrock serves Claude via cross-region inference profiles,
# so the defaults carry the "eu." prefix. Use the exact IDs your Bedrock
# console shows (Model catalog -> model detail -> inference profile ID).
PROD_MODEL = os.environ.get("BEDROCK_PROD_MODEL", "eu.anthropic.claude-haiku-4-5")
JUDGE_MODEL = os.environ.get("BEDROCK_JUDGE_MODEL", "eu.anthropic.claude-opus-4-8")

# --- Web app -----------------------------------------------------------------
# If set, the web UI and API require this password (login page / Bearer token).
# Leave unset for open access during local development.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# --- Datasets ----------------------------------------------------------------
# The whole stack (index, optimisation, evaluation, UI) runs on the active
# dataset. "nimbus" is the synthetic label-free KB; "golden" is a corpus +
# QA-with-gold-answers set fetched from Hugging Face (fetchable from the UI
# or via `python -m indexing.fetch_hf_golden`). Gold answers enable the
# extra correctness metric automatically.
DATASETS = {
    "nimbus": {"dir": ROOT / "data", "label": "Nimbus Desk (synthetic, label-free)"},
    "golden": {"dir": ROOT / "data" / "hf", "label": "Wikipedia golden (Hugging Face)"},
    "hotpotqa": {"dir": ROOT / "data" / "hotpotqa", "label": "HotpotQA multi-hop (golden)"},
}
DEFAULT_DATASET = os.environ.get("DATASET", "nimbus")

# Mutable at runtime via set_dataset() (used by the UI dataset switcher).
DATASET_ID: str
DATA_DIR: Path
KNOWLEDGE_BASE_PATH: Path
QUESTIONS_PATH: Path
COLLECTION_NAME: str


def set_dataset(dataset_id: str) -> None:
    global DATASET_ID, DATA_DIR, KNOWLEDGE_BASE_PATH, QUESTIONS_PATH, COLLECTION_NAME
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    DATASET_ID = dataset_id
    DATA_DIR = DATASETS[dataset_id]["dir"]
    KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
    QUESTIONS_PATH = DATA_DIR / "questions.json"
    COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION") or f"kb-{dataset_id}"


set_dataset(DEFAULT_DATASET if DEFAULT_DATASET in DATASETS else "nimbus")

# --- Retrieval ---------------------------------------------------------------
CHROMA_PATH = str(ROOT / ".chroma")
TOP_K = 3  # minimalistic retrieval, as in the demo: top-3, no re-ranking

# --- Paths -------------------------------------------------------------------
PROMPTS_DIR = ROOT / "prompts"
BASELINE_PROMPT_PATH = PROMPTS_DIR / "baseline.txt"
OPTIMIZED_PROMPT_PATH = PROMPTS_DIR / "optimized.txt"
RESULTS_DIR = ROOT / "results"


def dspy_lm(model_id: str, **kwargs):
    """Build a DSPy LM pointed at Bedrock via LiteLLM.

    drop_params lets LiteLLM silently drop sampling parameters that newer
    Claude models (Opus 4.7+, Sonnet 5) reject at the API level.
    """
    import dspy
    import litellm

    litellm.drop_params = True
    kwargs.setdefault("max_tokens", 1024)
    return dspy.LM(f"bedrock/{model_id}", aws_region_name=AWS_REGION, **kwargs)
