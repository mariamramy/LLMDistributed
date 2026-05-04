import argparse
import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from rag.retriever import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL


log = logging.getLogger("rag_ingest")

DEFAULT_PDF_CORPUS_DIR = "pdfs"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_BATCH_SIZE = 64


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    page: int
    chunk_index: int

    @property
    def metadata(self) -> dict:
        return {
            "source_file": self.source_file,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
        }


@dataclass
class IngestConfig:
    pdf_dir: Path
    chroma_host: str
    chroma_port: int
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    batch_size: int


def load_config() -> IngestConfig:
    return IngestConfig(
        pdf_dir=Path(os.getenv("PDF_CORPUS_DIR", DEFAULT_PDF_CORPUS_DIR)),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        collection_name=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
        batch_size=int(os.getenv("RAG_INGEST_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
    )


def find_pdfs(pdf_dir: Path) -> List[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def split_text(text: str, *, chunk_size: int, overlap: int) -> List[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = end - overlap
    return chunks


def make_chunk_id(source_file: str, page: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    safe_source = source_file.replace("/", "_").replace(" ", "_")
    return f"{safe_source}:p{page}:c{chunk_index}:{digest}"


def extract_pdf_chunks(
    pdf_path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[DocumentChunk]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    chunks: List[DocumentChunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk_index, chunk_text in enumerate(
            split_text(page_text, chunk_size=chunk_size, overlap=chunk_overlap),
            start=1,
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(pdf_path.name, page_index, chunk_index, chunk_text),
                    text=chunk_text,
                    source_file=pdf_path.name,
                    page=page_index,
                    chunk_index=chunk_index,
                )
            )
    return chunks


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


async def embed_texts(
    client: Any,
    texts: Sequence[str],
    *,
    model: str,
) -> List[List[float]]:
    response = await client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


async def ingest_chunks(
    collection: Any,
    openai_client: Any,
    chunks: Sequence[DocumentChunk],
    *,
    embedding_model: str,
    batch_size: int,
) -> int:
    inserted = 0
    for batch in batched(chunks, batch_size):
        documents = [chunk.text for chunk in batch]
        embeddings = await embed_texts(openai_client, documents, model=embedding_model)
        await asyncio.to_thread(
            collection.upsert,
            ids=[chunk.chunk_id for chunk in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in batch],
        )
        inserted += len(batch)
        log.info("Upserted %d/%d chunks", inserted, len(chunks))
    return inserted


async def run_ingestion(config: IngestConfig) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for PDF ingestion")

    pdfs = find_pdfs(config.pdf_dir)
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {config.pdf_dir}")

    chunks: List[DocumentChunk] = []
    for pdf_path in pdfs:
        pdf_chunks = extract_pdf_chunks(
            pdf_path,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        log.info("Extracted %d chunks from %s", len(pdf_chunks), pdf_path)
        chunks.extend(pdf_chunks)

    if not chunks:
        raise RuntimeError("PDF extraction produced no text chunks")

    import chromadb
    from openai import AsyncOpenAI

    chroma_client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
    collection = chroma_client.get_or_create_collection(name=config.collection_name)
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    inserted = await ingest_chunks(
        collection,
        openai_client,
        chunks,
        embedding_model=config.embedding_model,
        batch_size=config.batch_size,
    )
    log.info("Collection %s now contains %d chunks", config.collection_name, collection.count())
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local textbook PDFs into ChromaDB")
    parser.add_argument("--once", action="store_true", help="Run ingestion once and exit")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [RAG] %(levelname)s %(message)s")
    parse_args()
    inserted = asyncio.run(run_ingestion(load_config()))
    log.info("Ingestion complete, upserted %d chunks", inserted)


if __name__ == "__main__":
    main()
