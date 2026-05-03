import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from llm.inference import SourceSnippet


DEFAULT_COLLECTION_NAME = "distributed_systems_textbook"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 3


@dataclass
class RetrievalResult:
    sources: List[SourceSnippet]


class ChromaRAGRetriever:
    def __init__(
        self,
        *,
        openai_client: Any,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
        chroma_client: Optional[Any] = None,
    ):
        self.openai_client = openai_client
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "chromadb")
        self.chroma_port = int(chroma_port or os.getenv("CHROMA_PORT", "8000"))
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME
        )
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.top_k = int(top_k or os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K)))
        self._client = chroma_client
        self._collection = None
        self.retrieval_count = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            import chromadb

            self._client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(name=self.collection_name)
        return self._collection

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    async def count(self) -> int:
        return await asyncio.to_thread(self.collection.count)

    async def is_ready(self) -> bool:
        try:
            return await self.count() > 0
        except Exception:
            return False

    async def retrieve(self, prompt: str, top_k: Optional[int] = None) -> RetrievalResult:
        n_results = top_k or self.top_k
        embeddings = await self.embed_texts([prompt])
        result = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        self.retrieval_count += 1
        return RetrievalResult(sources=_parse_query_result(result))


def _parse_query_result(result: Dict[str, Any]) -> List[SourceSnippet]:
    documents = _first_query_row(result.get("documents"))
    metadatas = _first_query_row(result.get("metadatas"))
    distances = _first_query_row(result.get("distances"))

    sources: List[SourceSnippet] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = distances[index] if index < len(distances) else None
        sources.append(
            SourceSnippet(
                text=document or "",
                source_file=str(metadata.get("source_file", "")),
                page=int(metadata.get("page", 0) or 0),
                chunk_id=str(metadata.get("chunk_id", "")),
                score=_distance_to_score(distance),
            )
        )
    return sources


def _first_query_row(value: Any) -> List[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _distance_to_score(distance: Any) -> Optional[float]:
    if distance is None:
        return None
    try:
        numeric = float(distance)
    except (TypeError, ValueError):
        return None
    return max(0.0, 1.0 - numeric)


def retrieve_context(query: str) -> str:
    raise RuntimeError(
        "retrieve_context() has been replaced by ChromaRAGRetriever; use the worker HTTP service instead"
    )
