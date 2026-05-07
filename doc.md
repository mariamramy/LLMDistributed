# Worker/RAG/LLM Layer Documentation

## What This Layer Owns

This project section owns the worker side of the distributed LLM system:

- Three worker services: `worker-1`, `worker-2`, and `worker-3`
- Local Ollama-backed LLM generation
- Local Ollama-backed embeddings
- PDF textbook ingestion into ChromaDB
- RAG retrieval from ChromaDB
- Worker health, metrics, registration, and heartbeat behavior

The client, load balancer, and master scheduler are owned by the other team. Our worker API stays stable for that master node.

## Runtime Architecture

The Docker stack for our part contains:

- `ollama`: local model server on `localhost:11434`
- `ollama_pull`: one-shot job that pulls the generation and embedding models
- `chromadb`: shared vector database on `localhost:8000`
- `rag_ingest`: one-shot job that reads PDFs, embeds chunks, and writes them to ChromaDB
- `worker-1`: worker service exposed on `localhost:9101`
- `worker-2`: worker service exposed on `localhost:9102`
- `worker-3`: worker service exposed on `localhost:9103`

Inside Docker, all workers listen on port `9100`. Host ports differ only so we can test each worker directly.

Default local models:

```bash
OLLAMA_GENERATION_MODEL=llama3.2:3b-instruct-q3_K_M
OLLAMA_EMBEDDING_MODEL=all-minilm
```

This setup is intentionally small for older MacBook Air hardware. Keep `WORKER_MAX_CONCURRENCY=1` locally unless the machine has enough CPU/RAM headroom.

## Data Flow

### Ingestion Flow

1. The textbook PDF lives under `pdfs/`.
2. `ollama_pull` makes sure `llama3.2:3b-instruct-q3_K_M` and `all-minilm` are installed in the shared Ollama volume.
3. `rag_ingest` reads PDFs from `/app/pdfs`.
4. Text is extracted page by page with `pypdf`.
5. Page text is split into chunks.
6. Each chunk is embedded locally through Ollama `/api/embed`.
7. Chunks are upserted into ChromaDB collection `distributed_systems_textbook_ollama_all_minilm`.
8. The ingestion job exits successfully after indexing.

The ingestion job is idempotent because each chunk uses a deterministic ID. Re-running ingestion updates existing chunks instead of duplicating them.

### Request Flow

1. The master chooses a worker and sends `POST /task`.
2. The worker validates `task_id` and `prompt`.
3. If `use_rag=true`, the worker embeds only the user prompt through Ollama.
4. The worker queries ChromaDB for the top `RAG_TOP_K` chunks, default `3`.
5. The worker sends the user prompt plus those retrieved snippets to Ollama `/api/chat`.
6. The worker returns the answer, latency, model metadata, and source metadata.

Important: the full textbook is never placed into the generation request. ChromaDB stores the textbook chunks, and the worker sends only the few retrieved snippets needed to enrich the answer.

## APIs Exposed By Each Worker

### `GET /health`

Success response:

```json
{
  "worker_id": "worker-1",
  "status": "ok",
  "llm_provider": "ollama",
  "llm_ready": true,
  "rag_ready": true,
  "chroma_ready": true,
  "chunk_count": 2247,
  "active_tasks": 0,
  "total_tasks": 0,
  "load": 0.0,
  "model": "llama3.2:3b-instruct-q3_K_M",
  "embedding_model": "all-minilm"
}
```

Returns `503` when Ollama is unreachable, required models are missing, ChromaDB is unavailable, or no chunks are indexed.

### `GET /metrics`

```json
{
  "worker_id": "worker-1",
  "active_tasks": 1,
  "max_concurrency": 1,
  "load": 1.0,
  "total_tasks": 20,
  "completed_tasks": 18,
  "failed_tasks": 2,
  "avg_latency_ms": 850.0,
  "llm_provider": "ollama",
  "llm_ready": true,
  "rag_ready": true,
  "chroma_ready": true,
  "chunk_count": 2247,
  "retrieval_count": 18,
  "ollama_errors": 0,
  "chroma_errors": 0
}
```

### `POST /task`

Canonical request body:

```json
{
  "task_id": "task-123",
  "request_id": "client-request-123",
  "prompt": "Explain replication in distributed systems.",
  "use_rag": true
}
```

`query` is temporarily accepted as an alias for `prompt` to support older code.

Success response:

```json
{
  "task_id": "task-123",
  "request_id": "client-request-123",
  "status": "completed",
  "worker_id": "worker-1",
  "result": "answer text",
  "latency_ms": 1234.5,
  "rag": {
    "used": true,
    "sources": [
      {
        "source_file": "Distributed_Systems_4-230325.pdf",
        "page": 12,
        "chunk_id": "Distributed_Systems_4-230325.pdf:p12:c1:abc123",
        "score": 0.91
      }
    ]
  },
  "llm": {
    "model": "llama3.2:3b-instruct-q3_K_M",
    "usage": {
      "prompt_eval_count": 100,
      "eval_count": 80
    }
  }
}
```

Failure response:

```json
{
  "task_id": "task-123",
  "status": "failed",
  "worker_id": "worker-1",
  "error": {
    "code": "ollama_not_ready",
    "message": "Ollama is not reachable or required models are not installed"
  }
}
```

## Master Integration Contract

Workers can register and send heartbeats to the master when `MASTER_URL` is set.

Registration:

```http
POST {MASTER_URL}/register
```

```json
{
  "worker_id": "worker-1",
  "host": "worker-1",
  "port": 9100
}
```

Heartbeat:

```http
POST {MASTER_URL}/heartbeat
```

```json
{
  "worker_id": "worker-1",
  "load": 0.0,
  "active_tasks": 0,
  "max_concurrency": 1,
  "total_tasks": 20
}
```

The master should select workers using healthy status plus `(load, active_tasks)`.

## Configuration

Use `.env.example` as the template.

Important defaults:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_GENERATION_MODEL=llama3.2:3b-instruct-q3_K_M
OLLAMA_EMBEDDING_MODEL=all-minilm
OLLAMA_EMBED_MAX_CHARS=200
OLLAMA_MAX_OUTPUT_TOKENS=300
CHROMA_COLLECTION=distributed_systems_textbook_ollama_all_minilm
RAG_TOP_K=3
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=75
RAG_INGEST_BATCH_SIZE=16
WORKER_MAX_CONCURRENCY=1
HEARTBEAT_INTERVAL_S=5
MASTER_URL=
```

Set `MASTER_URL=http://scheduler:9000` when the master service exposes compatible `/register` and `/heartbeat` routes.

## Running The Stack

1. Put the textbook PDF in `pdfs/`.
2. Create `.env` from `.env.example` if you want to override defaults.
3. Run:

```bash
docker compose up --build
```

The first run downloads the official Ollama Docker image, then roughly 1.7 GB for the generation model plus about 46 MB for the embedding model. The image can include multi-GB platform layers, but after that Docker caches the image and models persist in the `ollama-data` Docker volume.

Useful checks:

```bash
curl http://localhost:11434/api/tags
curl http://localhost:8000/api/v2/heartbeat
curl http://localhost:9101/health
curl http://localhost:9101/metrics
```

Manual task:

```bash
curl -X POST http://localhost:9101/task \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "manual-1",
    "prompt": "Explain replication in distributed systems.",
    "use_rag": true
  }'
```

## Postman

Import these files into Postman:

- `postman/worker-rag-llm.postman_collection.json`
- `postman/worker-rag-llm.postman_environment.json`

Run the collection after `docker compose up --build` has finished model pull, ingestion, and worker startup.

The collection tests:

- Ollama model tags
- ChromaDB heartbeat
- Worker health
- Worker metrics
- RAG task execution
- No-RAG task execution
- Validation error for missing prompt
- All three worker services

## Tests

Local tests:

```bash
.venv/bin/python -m pytest -q
```

Compile check:

```bash
.venv/bin/python -m compileall -q workers rag llm tests
```

Compose validation:

```bash
docker compose config --quiet
```

Docker build check:

```bash
docker compose build rag_ingest worker-1
```

The unit tests use mocked Ollama and mocked ChromaDB where appropriate. They do not require live models.

## Files Added Or Changed

- `workers/gpu_workers.py`: aiohttp worker service, Ollama readiness, task handling, metrics
- `rag/ingest.py`: PDF-to-Chroma ingestion job using Ollama embeddings
- `rag/retriever.py`: Chroma-backed RAG retrieval using Ollama embeddings
- `llm/inference.py`: Ollama chat/embed wrapper and prompt construction
- `docker-compose.yml`: Ollama, model pull, ChromaDB, ingestion, placeholder services, and 3 workers
- `.env.example`: local Ollama defaults
- `requirements.txt`: runtime and test dependencies with no provider SDK requirement
- `postman/`: Postman collection and environment
- `tests/`: worker, ingestion, retrieval, and LLM prompt tests

## Operational Notes

- `pdfs/` is ignored by Git because textbooks can be large or copyrighted.
- `.env` is ignored by Git because it may contain local overrides.
- The worker stack can run before the master is ready; if `MASTER_URL` is empty, registration and heartbeat are skipped.
- ChromaDB collection `distributed_systems_textbook_ollama_all_minilm` is intentionally separate from the old embedding collection because vector dimensions and semantics differ by embedding model.
- The default chunk size is kept at 500 characters, and embedding inputs are capped to `OLLAMA_EMBED_MAX_CHARS=200` with `truncate=true`, so unusual PDF chunks cannot crash ingestion by exceeding the local embedding model context window.
- Live ingestion and live task execution use local Ollama instead of a third-party LLM provider.
