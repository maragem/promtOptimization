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
| `AWS_REGION` | Region where the Claude models are enabled (default `eu-west-1`). |
| `BEDROCK_PROD_MODEL` | Default `eu.anthropic.claude-haiku-4-5`. |
| `BEDROCK_JUDGE_MODEL` | Default `eu.anthropic.claude-opus-4-8`. |

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

## Web UI (eUI / EC look and feel)

A FastAPI app in `app/` serves an EC-styled single-page interface for the
assistant: ask questions (or click the sample chips — orange ones are
hallucination traps), switch between the **baseline** and **GEPA-optimised**
system prompt, inspect the retrieved context, and optionally score each
answer live with the relevancy/groundedness judges.

```bash
uvicorn app.main:app --reload
# open http://localhost:8000
```

Nothing is loaded at startup: after signing in, click **"Load model &
build index"** — initialisation downloads the embedding model (first run),
builds the Chroma index, and makes one test call to the LLM, so any
configuration problem surfaces there with the exact error before questions
are asked. The "Optimised (GEPA)" toggle activates once
`prompts/optimized.txt` exists (commit it after running the optimiser so
deployments include it).

### Access control

Set `APP_PASSWORD` to gate the UI and API behind a password. Browsers get an
EC-styled login page (`/login.html`) and a 12-hour HttpOnly session cookie;
scripts can authenticate with `Authorization: Bearer <password>`. The
`/api/health` endpoint stays open so Railway's healthcheck keeps working.
If `APP_PASSWORD` is unset (e.g. local development), the app is open.

## Deploying to Railway

The repo ships a `Dockerfile` and `railway.toml`, so deployment is:

1. Create a new Railway project → **Deploy from GitHub repo** → pick this
   repo/branch. Railway detects the Dockerfile automatically.
2. Under **Variables**, set:
   - `AWS_BEARER_TOKEN_BEDROCK` — your Bedrock API key
   - `AWS_REGION` — e.g. `eu-west-1`
   - `APP_PASSWORD` — access password for the UI/API (strongly recommended:
     without it, anyone with the URL can trigger Bedrock calls on your key)
   - optionally `BEDROCK_PROD_MODEL` / `BEDROCK_JUDGE_MODEL`
3. Add a public domain under **Settings → Networking**.

Notes:
- The first boot downloads the MiniLM embedding model and indexes the
  knowledge base; `railway.toml` sets a 300 s healthcheck timeout for this.
- Railway's filesystem is ephemeral — the Chroma index is rebuilt on each
  deploy from `data/knowledge_base.json`, which is fine at this size.
- Run the GEPA optimisation locally (it is an offline batch job), commit
  `prompts/optimized.txt`, and redeploy to expose it in the UI.

### Cost & call volume

GEPA is call-hungry: each iteration runs the pipeline over a batch and calls
both judges per example. `auto="light"` keeps this in the low hundreds of
calls; DSPy caches LLM responses on disk, so re-runs are cheap. Bump to
`auto="medium"` only once the light run looks sane.

## Golden datasets (Hugging Face)

The stack is dataset-agnostic: `DATASET_DIR` selects which corpus + question
set everything (indexing, optimisation, evaluation, UI) runs on. The default
is the label-free synthetic Nimbus KB. To use a golden dataset with reference
answers instead:

```bash
python -m indexing.fetch_hf_golden          # rag-datasets/rag-mini-wikipedia
export DATASET_DIR=data/hf
python -m indexing.build_index
python -m optimisation.run_gepa
```

The fetch script uses the Hugging Face datasets-server REST API (no extra
dependencies) and writes `data/hf/knowledge_base.json` + `questions.json`;
other datasets/columns are configurable via CLI flags. When questions carry
gold answers, a third LLM-judge metric — **correctness vs the reference
answer** — is added to relevancy + groundedness automatically, in both GEPA
and evaluation. Trade-off vs the deck's label-free approach: golden data
anchors the optimisation to ground truth, but ties you to a corpus that must
match the questions (that's why the fetch script pulls the dataset's own
passages as the knowledge base).

## Running the optimisation from the UI

The web app can run GEPA as a server-side background job: the **"Prompt
optimisation (GEPA)"** panel starts it (light budget), shows live progress
(judge evaluations, elapsed time, latest scores), and on completion displays
the winning prompt and activates the "Optimised (GEPA)" toggle. Endpoints:
`POST /api/optimize` and `GET /api/optimize/status`.

Caveats on Railway:
- the run must finish within one container lifetime (a redeploy kills it);
- the resulting `prompts/optimized.txt` lives on the ephemeral filesystem —
  copy it from the UI ("View the optimised prompt") into the repo and commit
  to make it permanent;
- one run at a time; a second start returns 409.

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

- **The UI hangs on "Calling the model…" / no `POST /api/ask` in the logs** —
  the Bedrock call itself is failing or blocked. Open
  `https://<your-app>/api/test-model` (while logged in): it makes one tiny
  Bedrock call and returns either the latency or the exact boto3 error.
  Typical fixes:
  - `ValidationException` mentioning on-demand throughput or an unknown
    model → your account serves Claude via an inference profile; set
    `BEDROCK_PROD_MODEL`/`BEDROCK_JUDGE_MODEL` to the region-prefixed ID
    shown in the Bedrock console (e.g. `eu.anthropic.…` / `us.anthropic.…`).
  - `AccessDeniedException` → enable the model in **Bedrock console →
    Model access** for that region, or fix `AWS_REGION`.
  - `NoCredentialsError` / `UnrecognizedClientException` → the
    `AWS_BEARER_TOKEN_BEDROCK` variable is missing or wrong on Railway.
  The Bedrock client now uses tight timeouts (10 s connect / 90 s read,
  2 attempts), so errors surface in the logs within seconds instead of
  hanging for minutes.

- **Validation error about `temperature`/`top_p` on the judge model** — newer
  Claude models (Opus 4.7+, Sonnet 5) reject sampling parameters. `config.py`
  sets `litellm.drop_params = True` to strip them; if your LiteLLM version
  still forwards them, set `BEDROCK_JUDGE_MODEL=anthropic.claude-sonnet-4-6`
  or upgrade LiteLLM.
- **`AccessDeniedException` / model not found** — the model isn't enabled in
  your region, or needs the inference-profile prefix (see Setup note).
- **Bearer token not picked up** — requires `boto3 >= 1.39`; check with
  `pip show boto3`.
