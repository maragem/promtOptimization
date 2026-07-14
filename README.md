# Prompt Optimisation — Haystack + DSPy/GEPA on Claude (Amazon Bedrock)

A runnable counterpart to the DIGIT "Prompt Optimisation example" presentation
(Redfield, 2026-07-03), rebuilt on:

- **Haystack** for the production RAG pipeline (indexing, retrieval, generation)
- **DSPy + GEPA** as an *offline* optimisation harness for the prompt
- **Claude models on Amazon Bedrock** (authenticated with a single Bedrock API
  key), designed so switching to **GPT@EC** later is a config change

The use case mirrors the deck: a RAG support assistant for an imaginary SaaS
("Nimbus Desk") whose naive starting prompt hallucinates when the retrieved
context does not fully support the question. GEPA evolves the prompt against
two LLM-as-a-judge metrics — **answer relevancy** and **groundedness** — with
no human-labelled data.

## Architecture

```
                 ┌────────────────────────────────────────────┐
                 │ PRODUCTION (Haystack)                       │
   question ───► │ Chroma retriever (top-3, MiniLM embeddings) │ ───► answer
                 │   -> ChatPromptBuilder (prompt from file)   │
                 │   -> Claude Haiku 4.5 on Bedrock            │
                 └───────────────▲────────────────────────────┘
                                 │ prompts/optimized.txt
                 ┌───────────────┴────────────────────────────┐
                 │ OFFLINE OPTIMISATION (DSPy)                 │
                 │ SupportRag mirror (same Chroma retriever)   │
                 │   + GEPA (judge = Claude Opus 4.8)          │
                 │   + metrics: relevancy & groundedness       │
                 └────────────────────────────────────────────┘
```

DSPy never runs in production. It optimises the prompt offline; the winning
instruction text is written to `prompts/optimized.txt`, which the Haystack
pipeline loads as its system prompt.

### Mapping to the presentation

| Deck component | This repo |
|---|---|
| ChromaDB + all-MiniLM-L6-v2, top-3 cosine | `ChromaDocumentStore` + `ChromaQueryTextRetriever` (Chroma's default EF *is* MiniLM) |
| "Production" model: llama-3.1-8b-instant | `anthropic.claude-haiku-4-5` (small/cheap tier, keeps the weak-model dynamic) |
| "Judge" model: Gemma | `anthropic.claude-opus-4-8` (strong judge + GEPA reflection model) |
| Answer relevancy + groundedness judges | `optimisation/metrics.py` |
| GEPA optimiser | `dspy.GEPA` in `optimisation/run_gepa.py` |
| 15 demo questions, no labels | `data/questions.json` (10 answerable, 5 hallucination traps) |
| Prompts + mean results slide | `evaluation/evaluate.py` before/after reports in `results/` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your Bedrock API key
```

Required environment (see `.env.example`):

| Variable | Purpose |
|---|---|
| `AWS_BEARER_TOKEN_BEDROCK` | Your Bedrock API key. boto3 (Haystack) and LiteLLM (DSPy) both read it automatically — no AWS access/secret keys needed. |
| `AWS_REGION` | Region where the Claude models are enabled (default `us-east-1`). |
| `BEDROCK_PROD_MODEL` | Default `anthropic.claude-haiku-4-5`. |
| `BEDROCK_JUDGE_MODEL` | Default `anthropic.claude-opus-4-8`. |

> If your Bedrock account serves Claude through cross-region inference
> profiles, prefix the model IDs with the region group, e.g.
> `eu.anthropic.claude-haiku-4-5`.

## Run the demo (the deck's storyline)

```bash
# 1. Index the knowledge base into Chroma
python -m indexing.build_index

# 2. Baseline: score the naive prompt through the real Haystack pipeline
python -m evaluation.evaluate --prompt prompts/baseline.txt

# 3. Optimise: GEPA evolves the prompt against the judges
python -m optimisation.run_gepa

# 4. After: score the optimised prompt and compare the means
python -m evaluation.evaluate --prompt prompts/optimized.txt
```

Reports land in `results/baseline.json` and `results/optimized.json`
(per-question answers, judge scores, and judge reasoning).

### Cost & call volume

GEPA is call-hungry: each iteration runs the pipeline over a batch and calls
both judges per example. `auto="light"` keeps this in the low hundreds of
calls; DSPy caches LLM responses on disk, so re-runs are cheap. Bump to
`auto="medium"` only once the light run looks sane.

## Switching to GPT@EC later

The stack is provider-agnostic by design; the swap is confined to two spots:

1. **DSPy side** (`config.dspy_lm`): LiteLLM speaks OpenAI-compatible APIs —
   ```python
   dspy.LM(f"openai/{model_id}", api_base=GPTEC_BASE_URL, api_key=os.environ["GPTEC_API_KEY"])
   ```
2. **Haystack side** (`rag/pipeline.py`): replace `AmazonBedrockChatGenerator`
   with `OpenAIChatGenerator(model=..., api_base_url=GPTEC_BASE_URL,
   api_key=Secret.from_env_var("GPTEC_API_KEY"))`.

Everything else — retrieval, prompts, metrics, GEPA, evaluation — is unchanged.
Keep the weak-production / strong-judge pairing when picking GPT@EC models.

## Troubleshooting

- **Validation error about `temperature`/`top_p` on the judge model** — newer
  Claude models (Opus 4.7+, Sonnet 5) reject sampling parameters. `config.py`
  sets `litellm.drop_params = True` to strip them; if your LiteLLM version
  still forwards them, set `BEDROCK_JUDGE_MODEL=anthropic.claude-sonnet-4-6`
  or upgrade LiteLLM.
- **`AccessDeniedException` / model not found** — the model isn't enabled in
  your region, or needs the inference-profile prefix (see Setup note).
- **Bearer token not picked up** — requires `boto3 >= 1.39`; check with
  `pip show boto3`.
