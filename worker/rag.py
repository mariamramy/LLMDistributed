from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any

import chromadb

from worker.embeddings import HashEmbeddingFunction


COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "knowledge_base")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "hash")


@lru_cache(maxsize=1)
def _get_sentence_embedder():
    if EMBEDDING_BACKEND != "sentence_transformers":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection() -> Any:
    client = chromadb.HttpClient(
        host=os.environ.get("CHROMA_HOST", "chromadb"),
        port=int(os.environ.get("CHROMA_PORT", "8000")),
    )
    embedder = _get_sentence_embedder()
    embedding_function = None if embedder else HashEmbeddingFunction()
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
    )


def _retrieve_context_sync(query: str, n_results: int) -> str:
    collection = _get_collection()
    embedder = _get_sentence_embedder()

    if embedder:
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )
    documents = results.get("documents") or []
    chunks = documents[0] if documents else []
    return "\n\n".join(chunks) if chunks else "No relevant context found."


async def retrieve_context(query: str, n_results: int = 3) -> str:
    try:
        return await asyncio.to_thread(_retrieve_context_sync, query, n_results)
    except Exception as exc:
        return f"No relevant context found. RAG retrieval failed: {exc}"
