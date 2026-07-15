"""Evaluate a prompt through the REAL Haystack pipeline (the production path).

Usage:
    python -m evaluation.evaluate                                   # baseline
    python -m evaluation.evaluate --prompt prompts/optimized.txt    # optimised

Runs every question in data/questions.json through the Haystack pipeline with
the given system prompt, scores each answer with the same LLM judges GEPA
used, prints per-question and mean scores, and saves a JSON report under
results/. This reproduces the deck's "prompts + mean results" comparison and
doubles as a regression suite for future prompt changes.
"""

import argparse
import json
import statistics
from pathlib import Path

import config
from optimisation.metrics import judge_answer
from rag.pipeline import answer, build_pipeline
from rag.retrieval import format_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default=str(config.BASELINE_PROMPT_PATH),
        help="Path to the prompt file to evaluate",
    )
    args = parser.parse_args()

    prompt_path = Path(args.prompt)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))

    pipe = build_pipeline()
    rows = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        reply, documents = answer(pipe, question, system_prompt)
        scores = judge_answer(
            question, format_context(documents), reply, gold_answer=item.get("answer")
        )
        row = {
            "question": question,
            "note": item.get("note", ""),
            "answer": reply,
            "relevancy": scores["relevancy"],
            "groundedness": scores["groundedness"],
            "feedback": scores["feedback"],
        }
        line = f"[{i:2}/{len(questions)}] rel={scores['relevancy']:.2f} grd={scores['groundedness']:.2f}"
        if "correctness" in scores:
            row["correctness"] = scores["correctness"]
            line += f" cor={scores['correctness']:.2f}"
        rows.append(row)
        print(f"{line} ({item.get('note', '')}) {question}")

    mean_rel = statistics.mean(r["relevancy"] for r in rows)
    mean_grd = statistics.mean(r["groundedness"] for r in rows)
    print("\n" + "=" * 60)
    print(f"Prompt:            {prompt_path}")
    print(f"Model:             {config.PROD_MODEL}")
    print(f"Mean relevancy:    {mean_rel:.3f}")
    print(f"Mean groundedness: {mean_grd:.3f}")
    correctness_rows = [r["correctness"] for r in rows if "correctness" in r]
    if correctness_rows:
        print(f"Mean correctness:  {statistics.mean(correctness_rows):.3f}")
    print("=" * 60)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    report_path = config.RESULTS_DIR / f"{prompt_path.stem}.json"
    report = {
        "prompt_file": str(prompt_path),
        "prompt": system_prompt,
        "model": config.PROD_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "mean_relevancy": mean_rel,
        "mean_groundedness": mean_grd,
        "results": rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
