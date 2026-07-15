# Project notes for Claude

## User preferences (apply to this and future projects)

- **AWS region: always use `eu-west-1`** as the default for anything AWS/Bedrock.
- In `eu-west-1`, Bedrock serves Claude models through **cross-region inference
  profiles** — model IDs must carry the `eu.` prefix (e.g.
  `eu.anthropic.claude-haiku-4-5`). Bare `anthropic.*` IDs fail with a
  ValidationException about on-demand throughput.
- Auth is a **Bedrock API key** (bearer token) in `AWS_BEARER_TOKEN_BEDROCK`
  — no AWS access/secret keys. Requires boto3 >= 1.39; LiteLLM reads the same
  env var.
- The eventual target LLM provider is **GPT@EC** (OpenAI-compatible API);
  keep provider-specific code confined to `config.py` and the generator in
  `rag/pipeline.py` so the switch stays a config change.

## Project shape

- `rag/` + `indexing/`: Haystack production RAG pipeline (Chroma, MiniLM via
  Chroma's default embedding function — deliberately no torch dependency).
- `optimisation/`: DSPy/GEPA offline prompt optimisation; winning prompt is
  written to `prompts/optimized.txt` and must be committed to reach deploys.
- `app/`: FastAPI + static frontend styled after eUI/EC look. Password gate
  via `APP_PASSWORD`; `/api/health` must stay public (Railway healthcheck).
- Deployment: Railway via Dockerfile; the deployed branch is configured in
  the Railway service settings (repo had no `main` as of 2026-07).
- Debugging model calls: `GET /api/test-model` makes one minimal Bedrock
  call and returns latency or the exact boto3 error.
