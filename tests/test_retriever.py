from types import SimpleNamespace

import pytest

from rag.retriever import ChromaRAGRetriever


class FakeEmbeddings:
    async def create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


class FakeCollection:
    def __init__(self):
        self.query_calls = []

    def count(self):
        return 2

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "documents": [["chunk a", "chunk b"]],
            "metadatas": [
                [
                    {"source_file": "book.pdf", "page": 10, "chunk_id": "a"},
                    {"source_file": "book.pdf", "page": 11, "chunk_id": "b"},
                ]
            ],
            "distances": [[0.1, 0.3]],
        }


class FakeChromaClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, name):
        return self.collection


@pytest.mark.asyncio
async def test_retriever_queries_chroma_with_prompt_embedding_and_top_k():
    collection = FakeCollection()
    retriever = ChromaRAGRetriever(
        openai_client=FakeOpenAIClient(),
        chroma_client=FakeChromaClient(collection),
        top_k=2,
    )

    result = await retriever.retrieve("replication")

    assert collection.query_calls[0]["query_embeddings"] == [[0.1, 0.2, 0.3]]
    assert collection.query_calls[0]["n_results"] == 2
    assert [source.text for source in result.sources] == ["chunk a", "chunk b"]
    assert result.sources[0].source_file == "book.pdf"
    assert result.sources[0].page == 10
    assert retriever.retrieval_count == 1
