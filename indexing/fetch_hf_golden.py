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
            batch = json.load(response)["rows"]
        rows.extend(r["row"] for r in batch)
        if len(batch) < length:
            break  # reached the end of the split
        offset += len(batch)
    return rows


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
    parser.add_argument("--out", default=str(config.ROOT / "data" / "hf"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching corpus from {args.dataset} ({args.corpus_config}/{args.corpus_split})...")
    corpus_rows = fetch_rows(args.dataset, args.corpus_config, args.corpus_split,
                             args.num_passages or None)
    knowledge_base = [
        {"id": f"hf-{i}", "content": str(row[args.corpus_column]).strip()}
        for i, row in enumerate(corpus_rows)
        if str(row.get(args.corpus_column, "")).strip()
    ]
    (out_dir / "knowledge_base.json").write_text(
        json.dumps(knowledge_base, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(knowledge_base)} passages -> {out_dir / 'knowledge_base.json'}")

    print(f"Fetching QA pairs ({args.qa_config}/{args.qa_split})...")
    qa_rows = fetch_rows(args.dataset, args.qa_config, args.qa_split, args.num_questions)
    questions = [
        {
            "question": str(row[args.question_column]).strip(),
            "answer": str(row[args.answer_column]).strip(),
            "note": "golden",
        }
        for row in qa_rows
        if str(row.get(args.question_column, "")).strip()
    ]
    (out_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(questions)} questions -> {out_dir / 'questions.json'}")

    print("\nNext steps:")
    print(f"  export DATASET_DIR={out_dir.relative_to(config.ROOT)}")
    print("  python -m indexing.build_index")
    print("  python -m optimisation.run_gepa   # now also optimises for correctness")


if __name__ == "__main__":
    main()
