"""Index the synthetic knowledge base into ChromaDB.

Usage:  python -m indexing.build_index
"""

import json

from haystack import Document

import config
from rag.retrieval import document_store


def main() -> None:
    entries = json.loads(config.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    documents = [Document(id=e["id"], content=e["content"]) for e in entries]

    store = document_store()
    store.write_documents(documents)
    print(f"Indexed {store.count_documents()} documents into '{config.COLLECTION_NAME}' at {config.CHROMA_PATH}")


if __name__ == "__main__":
    main()
