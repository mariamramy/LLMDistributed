import pytest

from rag.ingest import (
    DocumentChunk,
    delete_stale_chunks,
    ingest_chunks,
    make_chunk_id,
    split_text,
)


class FakeOllamaClient:
    def __init__(self):
        self.calls = []

    async def embed_texts(self, texts, *, model):
        self.calls.append({"texts": list(texts), "model": model})
        return [[float(index), 0.0] for index, _ in enumerate(texts)]


class FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class FakeCleanupCollection:
    def __init__(self):
        self.deleted_ids = []

    def get(self):
        return {"ids": ["id-1", "id-2", "stale-id"]}

    def delete(self, **kwargs):
        self.deleted_ids.extend(kwargs["ids"])


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
        FakeOllamaClient(),
        chunks,
        embedding_model="all-minilm",
        batch_size=10,
    )

    assert inserted == 2
    assert collection.upserts[0]["ids"] == ["id-1", "id-2"]
    assert collection.upserts[0]["metadatas"][0]["source_file"] == "book.pdf"
    assert collection.upserts[0]["documents"] == ["chunk one", "chunk two"]


@pytest.mark.asyncio
async def test_delete_stale_chunks_removes_ids_not_in_current_ingestion():
    collection = FakeCleanupCollection()

    deleted = await delete_stale_chunks(collection, ["id-1", "id-2"])

    assert deleted == 1
    assert collection.deleted_ids == ["stale-id"]
