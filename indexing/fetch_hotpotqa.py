"""Fetch HotpotQA (https://hotpotqa.github.io/) as a golden dataset.

HotpotQA is multi-hop: each question needs facts from TWO Wikipedia
paragraphs. Unlike rag-mini-wikipedia there is no separate corpus config —
every row embeds its own evidence paragraphs (in the "distractor" config:
2 gold + 8 distractors). The knowledge base is therefore built from the
context paragraphs of the sampled questions, deduplicated by title.

This makes it a deliberately hard benchmark for the demo's naive top-3
retrieval: both gold paragraphs must be retrieved for a grounded, correct
answer, so expect lower scores than on single-hop datasets — a good way to
show where prompt optimisation stops and retrieval quality starts.

Usage:
    python -m indexing.fetch_hotpotqa                 # 30 questions
    python -m indexing.fetch_hotpotqa --num-questions 50

Then:
    export DATASET=hotpotqa
    python -m indexing.build_index
"""

import argparse
import json
from pathlib import Path

import config
from indexing.fetch_hf_golden import fetch_rows

# The canonical HF id moved into the hotpotqa org; try old name as fallback.
DATASET_CANDIDATES = ("hotpotqa/hotpot_qa", "hotpot_qa")


def fetch_hotpotqa(
    out_dir: Path,
    num_questions: int = 30,
    hf_config: str = "distractor",
    split: str = "validation",
) -> tuple[int, int]:
    """Download HotpotQA rows and write knowledge_base.json / questions.json.

    Returns (number of passages, number of questions).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, last_error = None, None
    for dataset_name in DATASET_CANDIDATES:
        try:
            print(f"Fetching {dataset_name} ({hf_config}/{split})...")
            rows = fetch_rows(dataset_name, hf_config, split, num_questions)
            break
        except Exception as exc:  # try the alternate dataset id
            last_error = exc
    if rows is None:
        raise RuntimeError(f"Could not fetch HotpotQA from Hugging Face: {last_error}")

    passages: dict[str, str] = {}  # title -> paragraph text (deduplicated)
    questions = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        for title, sents in zip(titles, sentences):
            if title and title not in passages:
                text = " ".join(str(s).strip() for s in sents if str(s).strip())
                passages[title] = f"{title}: {text}"
        questions.append({"question": question, "answer": answer, "note": "golden"})

    knowledge_base = [
        {"id": f"hotpot-{i}", "content": text}
        for i, text in enumerate(passages.values())
    ]
    (out_dir / "knowledge_base.json").write_text(
        json.dumps(knowledge_base, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(knowledge_base)} passages -> {out_dir / 'knowledge_base.json'}")
    print(f"  {len(questions)} questions -> {out_dir / 'questions.json'}")
    return len(knowledge_base), len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--config", dest="hf_config", default="distractor",
                        choices=["distractor", "fullwiki"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", default=str(config.DATASETS["hotpotqa"]["dir"]))
    args = parser.parse_args()

    fetch_hotpotqa(
        Path(args.out),
        num_questions=args.num_questions,
        hf_config=args.hf_config,
        split=args.split,
    )

    print("\nNext steps:")
    print("  export DATASET=hotpotqa")
    print("  python -m indexing.build_index")
    print("  python -m optimisation.run_gepa")


if __name__ == "__main__":
    main()
