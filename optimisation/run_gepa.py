"""Run GEPA prompt optimisation and export the winning prompt for Haystack.

CLI usage:  python -m optimisation.run_gepa
Also used by the web app (POST /api/optimize) via run_optimisation().

Reads prompts/baseline.txt as the starting (parent) prompt, evolves it with
GEPA against the LLM judges (relevancy + groundedness, plus correctness when
the dataset carries gold answers), and writes the optimised instruction text
to prompts/optimized.txt — which the Haystack pipeline loads as its system
prompt.
"""

import json
import random

import dspy

import config
from optimisation.dspy_program import SupportRag
from optimisation.metrics import judge_answer, judge_lm


def load_examples() -> list:
    """Build DSPy examples from the active dataset (gold answers optional)."""
    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    examples = []
    for q in questions:
        fields = {"question": q["question"]}
        if q.get("answer"):
            fields["answer"] = q["answer"]
        examples.append(dspy.Example(**fields).with_inputs("question"))
    return examples


def run_optimisation(budget: str = "light", progress=None) -> str:
    """Run GEPA and return the optimised prompt (also written to disk).

    `progress`, if given, is called with the judge-result dict after every
    metric evaluation — used by the web UI for live status. It may be called
    from worker threads.
    """
    examples = load_examples()
    random.Random(0).shuffle(examples)
    split = max(1, (len(examples) * 2) // 3)
    trainset, valset = examples[:split], examples[split:] or examples[:1]

    baseline_prompt = config.BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    program = SupportRag(baseline_prompt)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        result = judge_answer(
            gold.question, pred.context, pred.answer,
            gold_answer=getattr(gold, "answer", None),
        )
        if progress is not None:
            try:
                progress(result)
            except Exception:
                pass  # progress reporting must never break the optimisation
        return dspy.Prediction(score=result["score"], feedback=result["feedback"])

    with dspy.context(lm=config.dspy_lm(config.PROD_MODEL)):
        optimizer = dspy.GEPA(
            metric=metric,
            auto=budget,  # "light" | "medium" | "heavy"
            reflection_lm=judge_lm(),  # the strong model proposes the child prompts
            track_stats=True,
        )
        optimized = optimizer.compile(program, trainset=trainset, valset=valset)

    optimized_prompt = optimized.generate.signature.instructions.strip()
    config.OPTIMIZED_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.OPTIMIZED_PROMPT_PATH.write_text(optimized_prompt + "\n", encoding="utf-8")
    return optimized_prompt


def main() -> None:
    baseline_prompt = config.BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    optimized_prompt = run_optimisation()

    print("\n" + "=" * 72)
    print("Baseline prompt:\n" + baseline_prompt)
    print("-" * 72)
    print("Optimised prompt:\n" + optimized_prompt)
    print("=" * 72)
    print(f"\nWritten to {config.OPTIMIZED_PROMPT_PATH}")
    print("Next: python -m evaluation.evaluate --prompt prompts/optimized.txt")


if __name__ == "__main__":
    main()
