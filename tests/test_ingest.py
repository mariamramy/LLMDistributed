from types import SimpleNamespace

import pytest

from rag.ingest import (
    DocumentChunk,
    ingest_chunks,
    make_chunk_id,
    split_text,
)


class FakeEmbeddings:
    async def create(self, model, input):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(index), 0.0]) for index, _ in enumerate(input)]
        )


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


class FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


def test_split_text_uses_overlap_and_never_returns_full_empty_text():
    chunks = split_text("abcdefghij", chunk_size=4, overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]
    assert split_text("   ", chunk_size=4, overlap=1) == []


def test_make_chunk_id_is_deterministic():
    first = make_chunk_id("book.pdf", 1, 2, "same text")
    second = make_chunk_id("book.pdf", 1, 2, "same text")

    assert first == second


@pytest.mark.asyncio
async def test_ingest_chunks_upserts_deterministic_ids_without_duplicates():
    collection = FakeCollection()
    chunks = [
        DocumentChunk("id-1", "chunk one", "book.pdf", 1, 1),
        DocumentChunk("id-2", "chunk two", "book.pdf", 1, 2),
    ]

    inserted = await ingest_chunks(
        collection,
        FakeOpenAIClient(),
        chunks,
        embedding_model="text-embedding-3-small",
        batch_size=10,
    )

    assert inserted == 2
    assert collection.upserts[0]["ids"] == ["id-1", "id-2"]
    assert collection.upserts[0]["metadatas"][0]["source_file"] == "book.pdf"
    assert collection.upserts[0]["documents"] == ["chunk one", "chunk two"]
