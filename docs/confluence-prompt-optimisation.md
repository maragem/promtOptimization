# Prompt Optimisation — What It Is and How It Works

## What is prompt optimisation?

Prompt optimisation is the **systematic, metric-driven improvement of the
instructions given to a large language model**, replacing manual
"prompt engineering" by trial and error.

Instead of a developer repeatedly rewording a prompt and eyeballing the
results, an optimisation algorithm runs the AI system on a batch of
questions, **measures** the quality of every answer against defined metrics,
and **automatically rewrites the prompt** to score better — keeping a change
only when it is a proven improvement.

The prompt is treated like code: generated, tested, versioned, and only
deployed when it demonstrably outperforms the previous one.

## Why it matters

- **Fewer hallucinations** — the strongest lever for making a
  retrieval-augmented (RAG) assistant admit "that information is not in my
  sources" instead of inventing an answer.
- **No labelled data required** — quality can be judged by a second LLM
  ("LLM-as-a-judge"), so any service can apply this to its own documents
  without building a ground-truth dataset first.
- **Lower cost, faster delivery** — prompt tuning stops consuming developer
  time, and an optimised prompt often lets a smaller, cheaper model match a
  larger one.
- **Model- and vendor-agnostic** — the method transfers unchanged between
  providers (e.g. Claude on Amazon Bedrock today, GPT@EC tomorrow).

## How it is achieved

Three building blocks are needed:

### 1. Metrics — defining "better"

Each answer is scored 0–1 by a strong judge model:

| Metric | Question it answers | Needs labels? |
|---|---|---|
| **Answer relevancy** | Does the answer actually address the question? | No |
| **Groundedness** | Is every claim supported by the retrieved sources? | No |
| **Correctness** | Does the answer match a gold reference answer? | Yes (optional) |

Relevancy and groundedness are deliberately paired: groundedness alone could
be gamed by always refusing to answer; relevancy punishes that.

### 2. An optimisation algorithm — GEPA (Genetic-Pareto)

1. **Run** the system on a batch of questions with the current *parent* prompt.
2. **Score** every answer with the metrics; the judge also writes textual
   feedback explaining each score.
3. **Reflect** — a strong LLM reads the feedback and proposes improved
   *child* prompts.
4. **Select** — a child replaces the parent only if it improves on at least
   one question **without regressing on any other** (the Pareto rule).
5. **Repeat** until the score plateaus or the budget is spent.

The output is simply text — an improved system prompt that is committed to
version control and loaded by the production pipeline. The optimiser never
runs in production.

### 3. Evaluation data

Either a plain list of representative user questions (label-free — the
judges provide the signal), or a *golden dataset* with reference answers
(e.g. from Hugging Face) when ground truth is available and the extra
correctness metric is wanted.

## What a result looks like

- **Before (naive prompt):** "You are a friendly support assistant. Use the
  context to answer the question." → fluent answers, but confidently invents
  product features when the sources don't cover the question.
- **After (optimised prompt):** explicit rules about sticking to the
  provided sources, stating when information is unavailable, and pointing
  the user to a next step → groundedness rises sharply, hallucinations
  largely disappear.

The same judge metrics then remain in place as a **regression suite**: every
future prompt change is scored before it ships.

## Our demonstrator

A working implementation is available: a Haystack RAG pipeline (ChromaDB +
MiniLM retrieval), DSPy/GEPA optimisation with Claude models on Amazon
Bedrock (small model in production, strong model as judge), three switchable
datasets (synthetic label-free, Wikipedia golden, HotpotQA multi-hop), and a
web UI where the optimisation can be launched and the before/after scores
inspected live. The provider layer is isolated so moving to **GPT@EC** is a
configuration change.

*Repository: `maragem/promtoptimization` — see the README for setup and the
Railway deployment.*

## Limits to keep in mind

- Prompt optimisation fixes **prompt** problems; if retrieval misses the
  evidence (e.g. multi-hop questions), no prompt can recover it — that is a
  retrieval-quality workstream.
- Judge models have blind spots; for high-stakes uses, complement them with
  golden data or human spot-checks.
- Optimisation runs cost real model calls (hundreds per run) — it is an
  offline batch activity, not something done per request.
