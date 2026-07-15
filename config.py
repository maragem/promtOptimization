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

# --- Dataset -----------------------------------------------------------------
# DATASET_DIR selects the corpus + questions the whole stack runs on:
#   data     -> synthetic Nimbus Desk KB, label-free questions (default)
#   data/hf  -> golden dataset fetched from Hugging Face
#               (python -m indexing.fetch_hf_golden), questions carry gold
#               answers that enable the extra correctness metric.
DATA_DIR = ROOT / os.environ.get("DATASET_DIR", "data")

# --- Retrieval ---------------------------------------------------------------
CHROMA_PATH = str(ROOT / ".chroma")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", f"kb-{DATA_DIR.name}")
TOP_K = 3  # minimalistic retrieval, as in the demo: top-3, no re-ranking

# --- Paths -------------------------------------------------------------------
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
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
