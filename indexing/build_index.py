"""Index the active dataset's knowledge base into ChromaDB.

Usage:  python -m indexing.build_index
"""

import json

from haystack import Document

import config
from rag.retrieval import document_store

BATCH_SIZE = 50  # embed in small batches to keep memory flat on small containers


def build_index(progress=None) -> int:
    """(Re)build the index for the active dataset. Returns the document count.

    `progress`, if given, is called with (indexed_so_far, total) after each
    batch — used by the web UI for live status.
    """
    entries = json.loads(config.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    documents = [Document(id=e["id"], content=e["content"]) for e in entries]

    store = document_store()
    for i in range(0, len(documents), BATCH_SIZE):
        store.write_documents(documents[i : i + BATCH_SIZE])
        if progress is not None:
            progress(min(i + BATCH_SIZE, len(documents)), len(documents))
    return store.count_documents()


def main() -> None:
    count = build_index(progress=lambda done, total: print(f"  indexed {done}/{total}"))
    print(f"Indexed {count} documents into '{config.COLLECTION_NAME}' at {config.CHROMA_PATH}")


if __name__ == "__main__":
    main()
