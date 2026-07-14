"""Shared retrieval layer.

Both the Haystack production pipeline and the DSPy optimisation harness pull
documents through this module, so retrieval is byte-identical in both worlds.

Embeddings use Chroma's built-in default embedding function, which is
all-MiniLM-L6-v2 (ONNX) — the same embedding model as the original demo,
without a heavyweight torch dependency.
"""

from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

import config


def document_store() -> ChromaDocumentStore:
    return ChromaDocumentStore(
        collection_name=config.COLLECTION_NAME,
        persist_path=config.CHROMA_PATH,
    )


def retriever(top_k: int = config.TOP_K) -> ChromaQueryTextRetriever:
    return ChromaQueryTextRetriever(document_store=document_store(), top_k=top_k)


def retrieve_documents(question: str, top_k: int = config.TOP_K):
    return retriever(top_k).run(query=question)["documents"]


def format_context(documents) -> str:
    return "\n\n".join(f"[{i + 1}] {doc.content}" for i, doc in enumerate(documents))
