"""Fetch a golden RAG dataset (corpus + QA pairs) from Hugging Face.

Uses the public datasets-server REST API, so no extra dependencies are
needed. The default dataset is rag-datasets/rag-mini-wikipedia, which ships
both a passage corpus and question-answer pairs with gold answers.

Usage:
    python -m indexing.fetch_hf_golden                # defaults, 30 questions
    python -m indexing.fetch_hf_golden --num-questions 50

Then point the stack at the golden dataset and rebuild the index:
    export DATASET_DIR=data/hf
    python -m indexing.build_index

Gold answers in questions.json enable the third metric (answer correctness
vs. the reference) in optimisation and evaluation automatically.
"""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import config

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # datasets-server maximum page size


def fetch_rows(dataset: str, config_name: str, split: str, limit: int | None):
    rows, offset = [], 0
    while limit is None or len(rows) < limit:
        length = PAGE if limit is None else min(PAGE, limit - len(rows))
        params = urllib.parse.urlencode(
            {"dataset": dataset, "config": config_name, "split": split,
             "offset": offset, "length": length}
        )
        with urllib.request.urlopen(f"{API}?{params}", timeout=60) as response:
            payload = json.load(response)
        if "rows" not in payload:
            raise RuntimeError(
                f"datasets-server returned no rows for {dataset} "
                f"(config={config_name}, split={split}): {payload}"
            )
        batch = payload["rows"]
        rows.extend(r["row"] for r in batch)
        if len(batch) < length:
            break  # reached the end of the split
        offset += len(batch)
    return rows


def _resolve_column(rows: list, preferred: str, fallbacks: tuple[str, ...]) -> str:
    """Use the preferred column if present; otherwise fall back gracefully."""
    if not rows:
        raise RuntimeError("Dataset returned no rows")
    sample = rows[0]
    for name in (preferred, *fallbacks):
        if isinstance(sample.get(name), str):
            return name
    for name, value in sample.items():  # last resort: first string column
        if isinstance(value, str):
            return name
    raise RuntimeError(f"No text column found; available columns: {list(sample)}")


def fetch_golden(
    out_dir: Path,
    dataset: str = "rag-datasets/rag-mini-wikipedia",
    corpus_config: str = "text-corpus",
    corpus_split: str = "passages",
    corpus_column: str = "passage",
    qa_config: str = "question-answer",
    qa_split: str = "test",
    question_column: str = "question",
    answer_column: str = "answer",
    num_passages: int = 0,
    num_questions: int = 30,
) -> tuple[int, int]:
    """Download corpus + QA pairs and write knowledge_base.json / questions.json.

    Returns (number of passages, number of questions). Also used by the web
    app's dataset switcher.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching corpus from {dataset} ({corpus_config}/{corpus_split})...")
    corpus_rows = fetch_rows(dataset, corpus_config, corpus_split, num_passages or None)
    corpus_column = _resolve_column(corpus_rows, corpus_column, ("passage", "text", "content"))
    knowledge_base = [
        {"id": f"hf-{i}", "content": str(row[corpus_column]).strip()}
        for i, row in enumerate(corpus_rows)
        if str(row.get(corpus_column, "")).strip()
    ]
    (out_dir / "knowledge_base.json").write_text(
        json.dumps(knowledge_base, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(knowledge_base)} passages -> {out_dir / 'knowledge_base.json'}")

    print(f"Fetching QA pairs ({qa_config}/{qa_split})...")
    qa_rows = fetch_rows(dataset, qa_config, qa_split, num_questions)
    question_column = _resolve_column(qa_rows, question_column, ("question", "query"))
    answer_column = _resolve_column(qa_rows, answer_column, ("answer", "answers", "response"))
    questions = [
        {
            "question": str(row[question_column]).strip(),
            "answer": str(row[answer_column]).strip(),
            "note": "golden",
        }
        for row in qa_rows
        if str(row.get(question_column, "")).strip()
    ]
    (out_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(questions)} questions -> {out_dir / 'questions.json'}")
    return len(knowledge_base), len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rag-datasets/rag-mini-wikipedia")
    parser.add_argument("--corpus-config", default="text-corpus")
    parser.add_argument("--corpus-split", default="passages")
    parser.add_argument("--corpus-column", default="passage")
    parser.add_argument("--qa-config", default="question-answer")
    parser.add_argument("--qa-split", default="test")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--num-passages", type=int, default=0,
                        help="0 = the full corpus (recommended: subsetting the "
                             "corpus orphans questions whose evidence is cut)")
    parser.add_argument("--num-questions", type=int, default=30)
    parser.add_argument("--out", default=str(config.DATASETS["golden"]["dir"]))
    args = parser.parse_args()

    fetch_golden(
        Path(args.out),
        dataset=args.dataset,
        corpus_config=args.corpus_config,
        corpus_split=args.corpus_split,
        corpus_column=args.corpus_column,
        qa_config=args.qa_config,
        qa_split=args.qa_split,
        question_column=args.question_column,
        answer_column=args.answer_column,
        num_passages=args.num_passages,
        num_questions=args.num_questions,
    )

    print("\nNext steps:")
    print("  export DATASET=golden")
    print("  python -m indexing.build_index")
    print("  python -m optimisation.run_gepa   # now also optimises for correctness")


if __name__ == "__main__":
    main()
