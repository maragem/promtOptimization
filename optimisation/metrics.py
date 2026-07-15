"""LLM-as-a-judge metrics: answer relevancy + groundedness.

Same design as the demo deck:
- Relevancy (question + answer only) stops the optimiser from gaming the
  process by instructing the model to always refuse.
- Groundedness (question + context + answer) is what actually reduces
  hallucination.

For GEPA the combined metric returns a score AND textual feedback — GEPA
uses the feedback to propose better child prompts.
"""

import dspy

import config

_judge_lm = None


def judge_lm():
    global _judge_lm
    if _judge_lm is None:
        _judge_lm = config.dspy_lm(config.JUDGE_MODEL)
    return _judge_lm


class RelevancyJudge(dspy.Signature):
    """Rate from 0.0 to 1.0 how relevant and helpful the answer is as a
    response to the customer's question. Judge only the question-answer fit —
    you do not know what source material was available.

    An answer that honestly states the requested information is unavailable
    and offers a concrete next step (e.g. contacting support) is still a
    relevant, helpful answer. An answer that dodges the question, is empty,
    or refuses without pointing anywhere scores low."""

    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Brief justification for the score")
    score: float = dspy.OutputField(desc="A number between 0.0 and 1.0")


class GroundednessJudge(dspy.Signature):
    """Rate from 0.0 to 1.0 whether every factual claim in the answer is
    supported by the provided context passages.

    1.0 means all claims are directly supported. Deduct for any claim about
    the product, its features, pricing, or policies that does not appear in
    the context (hallucination), even if plausible. A statement that the
    information is not available is grounded if and only if the context
    really does not contain it. Generic politeness carries no penalty."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Brief justification, naming any unsupported claims")
    score: float = dspy.OutputField(desc="A number between 0.0 and 1.0")


class CorrectnessJudge(dspy.Signature):
    """Rate from 0.0 to 1.0 whether the answer agrees with the gold reference
    answer. 1.0 means factually equivalent (wording may differ, extra correct
    detail is fine); 0.0 means it contradicts or misses the reference. If the
    answer claims the information is unavailable although a gold answer
    exists, score 0.0."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField(desc="The reference (gold) answer")
    answer: str = dspy.InputField(desc="The answer being evaluated")
    reasoning: str = dspy.OutputField(desc="Brief justification for the score")
    score: float = dspy.OutputField(desc="A number between 0.0 and 1.0")


def _clamp(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def judge_answer(question: str, context: str, answer: str, gold_answer: str | None = None) -> dict:
    """Score one (question, context, answer) triple.

    Always scores relevancy + groundedness (label-free). When a gold
    reference answer is available (golden datasets), also scores correctness
    and folds it into the combined score.
    """
    with dspy.context(lm=judge_lm()):
        rel = dspy.Predict(RelevancyJudge)(question=question, answer=answer)
        grd = dspy.Predict(GroundednessJudge)(question=question, context=context, answer=answer)
        cor = None
        if gold_answer:
            cor = dspy.Predict(CorrectnessJudge)(
                question=question, gold_answer=gold_answer, answer=answer
            )

    scores = {"relevancy": _clamp(rel.score), "groundedness": _clamp(grd.score)}
    feedback_lines = [
        f"Answer relevancy: {scores['relevancy']:.2f} — {rel.reasoning}",
        f"Groundedness: {scores['groundedness']:.2f} — {grd.reasoning}",
    ]
    if cor is not None:
        scores["correctness"] = _clamp(cor.score)
        feedback_lines.append(
            f"Correctness vs gold answer: {scores['correctness']:.2f} — {cor.reasoning}"
        )

    return {
        **scores,
        "score": sum(scores.values()) / len(scores),
        "feedback": "\n".join(feedback_lines),
    }


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric: mean of the judge scores + rich feedback."""
    result = judge_answer(
        gold.question, pred.context, pred.answer,
        gold_answer=getattr(gold, "answer", None),
    )
    return dspy.Prediction(score=result["score"], feedback=result["feedback"])
