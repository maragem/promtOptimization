"""Run GEPA prompt optimisation and export the winning prompt for Haystack.

Usage:  python -m optimisation.run_gepa

Reads prompts/baseline.txt as the starting (parent) prompt, evolves it with
GEPA against the relevancy + groundedness judges, and writes the optimised
instruction text to prompts/optimized.txt — which the Haystack pipeline can
then load as its system prompt (see evaluation/evaluate.py --prompt).
"""

import json
import random

import dspy

import config
from optimisation.dspy_program import SupportRag
from optimisation.metrics import gepa_metric, judge_lm


def main() -> None:
    dspy.configure(lm=config.dspy_lm(config.PROD_MODEL))

    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    examples = [
        dspy.Example(question=q["question"]).with_inputs("question") for q in questions
    ]
    # No human labels needed — the judges provide the training signal.
    random.Random(0).shuffle(examples)
    trainset, valset = examples[:10], examples[10:]

    baseline_prompt = config.BASELINE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    program = SupportRag(baseline_prompt)

    optimizer = dspy.GEPA(
        metric=gepa_metric,
        auto="light",  # bump to "medium"/"heavy" for more thorough (and costly) search
        reflection_lm=judge_lm(),  # the strong model proposes the child prompts
        track_stats=True,
    )
    optimized = optimizer.compile(program, trainset=trainset, valset=valset)

    optimized_prompt = optimized.generate.signature.instructions.strip()
    config.OPTIMIZED_PROMPT_PATH.write_text(optimized_prompt + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("Baseline prompt:\n" + baseline_prompt)
    print("-" * 72)
    print("Optimised prompt:\n" + optimized_prompt)
    print("=" * 72)
    print(f"\nWritten to {config.OPTIMIZED_PROMPT_PATH}")
    print("Next: python -m evaluation.evaluate --prompt prompts/optimized.txt")


if __name__ == "__main__":
    main()
