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
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_REGION)

# If your account requires cross-region inference profiles, prefix the IDs
# with the region group, e.g. "eu.anthropic.claude-haiku-4-5".
PROD_MODEL = os.environ.get("BEDROCK_PROD_MODEL", "anthropic.claude-haiku-4-5")
JUDGE_MODEL = os.environ.get("BEDROCK_JUDGE_MODEL", "anthropic.claude-opus-4-8")

# --- Web app -----------------------------------------------------------------
# If set, the web UI and API require this password (login page / Bearer token).
# Leave unset for open access during local development.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# --- Retrieval ---------------------------------------------------------------
CHROMA_PATH = str(ROOT / ".chroma")
COLLECTION_NAME = "nimbus_kb"
TOP_K = 3  # minimalistic retrieval, as in the demo: top-3, no re-ranking

# --- Paths -------------------------------------------------------------------
DATA_DIR = ROOT / "data"
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
