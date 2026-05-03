# Worker/RAG/LLM Layer Documentation

## What This Layer Owns

This project section owns the worker side of the distributed LLM system:

- Three worker services: `worker-1`, `worker-2`, and `worker-3`
- PDF textbook ingestion into ChromaDB
- RAG retrieval from ChromaDB
- OpenAI answer generation
- Worker health, metrics, registration, and heartbeat behavior

The client, load balancer, and master scheduler are owned by the other team. Our workers are ready for that master node to call.

## Runtime Architecture

The Docker stack for our part contains:

- `chromadb`: shared vector database on `localhost:8000`
- `rag_ingest`: one-shot job that reads PDFs, embeds chunks, and writes them to ChromaDB
- `worker-1`: worker service exposed on `localhost:9101`
- `worker-2`: worker service exposed on `localhost:9102`
- `worker-3`: worker service exposed on `localhost:9103`

Inside Docker, all workers listen on port `9100`. Host ports are different only so we can test each worker directly from the host.

## Data Flow

### Ingestion Flow

1. The textbook PDF lives under `pdfs/`.
2. `rag_ingest` reads the PDF files from `/app/pdfs`.
3. Text is extracted page by page with `pypdf`.
4. Page text is split into chunks.
5. Each chunk is embedded with OpenAI `text-embedding-3-small`.
6. Chunks are upserted into the ChromaDB collection `distributed_systems_textbook`.
7. The ingestion job exits successfully after indexing.

The ingestion job is idempotent because each chunk uses a deterministic ID. Re-running ingestion updates existing chunks instead of creating duplicate IDs.

### Request Flow

1. The master chooses a worker and sends `POST /task`.
2. The worker validates `task_id` and `prompt`.
3. If `use_rag=true`, the worker embeds only the user prompt.
4. The worker queries ChromaDB for the top `RAG_TOP_K` chunks, default `3`.
5. The worker sends the user prompt plus those retrieved snippets to OpenAI.
6. The worker returns the answer, latency, model metadata, and source metadata.

Important: the full textbook is not sent to OpenAI on every request. Only retrieved snippets are added to the generation prompt.

## APIs Exposed By Each Worker

### `GET /health`

Used by humans, Docker checks, and future orchestration.

Success response:

```json
{
  "worker_id": "worker-1",
  "status": "ok",
  "openai_configured": true,
  "rag_ready": true,
  "chroma_ready": true,
  "chunk_count": 240,
  "active_tasks": 0,
  "total_tasks": 0,
  "load": 0.0,
  "model": "gpt-5.4-nano"
}
```

Returns `503` when OpenAI is not configured, ChromaDB is unavailable, or no chunks are indexed.

### `GET /metrics`

Used for monitoring worker behavior.

```json
{
  "worker_id": "worker-1",
  "active_tasks": 1,
  "max_concurrency": 4,
  "load": 0.25,
  "total_tasks": 20,
  "completed_tasks": 18,
  "failed_tasks": 2,
  "avg_latency_ms": 850.0,
  "rag_ready": true,
  "chroma_ready": true,
  "chunk_count": 240,
  "retrieval_count": 18,
  "openai_errors": 0,
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
    "model": "gpt-5.4-nano",
    "usage": {}
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
    "code": "missing_prompt",
    "message": "prompt is required"
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
  "load": 0.25,
  "active_tasks": 1,
  "max_concurrency": 4,
  "total_tasks": 20
}
```

The master should select workers using healthy status plus `(load, active_tasks)`.

## Configuration

Use `.env.example` as the template.

Required:

```bash
OPENAI_API_KEY=...
```

Important defaults:

```bash
OPENAI_MODEL=gpt-5.4-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=3
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
WORKER_MAX_CONCURRENCY=4
HEARTBEAT_INTERVAL_S=5
MASTER_URL=
```

Set `MASTER_URL=http://scheduler:9000` when the master service exists in Compose.

## Running The Stack

1. Put the textbook PDF in `pdfs/`.
2. Create `.env` from `.env.example`.
3. Set `OPENAI_API_KEY`.
4. Run:

```bash
docker compose up --build
```

Useful checks:

```bash
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

Run the collection after `docker compose up --build` has finished ingestion and the workers are healthy.

The collection tests:

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
docker compose build rag_ingest
```

The unit tests use mocked OpenAI and mocked ChromaDB where appropriate. They do not spend OpenAI credits.

## Files Added Or Changed

- `workers/gpu_workers.py`: aiohttp worker service
- `rag/ingest.py`: PDF-to-Chroma ingestion job
- `rag/retriever.py`: Chroma-backed RAG retrieval
- `llm/inference.py`: OpenAI Responses API wrapper and prompt construction
- `docker-compose.yml`: ChromaDB, ingestion, and 3 workers
- `requirements.txt`: runtime and test dependencies
- `postman/`: Postman collection and environment
- `tests/`: worker, ingestion, retrieval, and LLM prompt tests

## Operational Notes

- `pdfs/` is ignored by Git because textbooks can be large or copyrighted.
- `.env` is ignored by Git because it contains secrets.
- The worker stack can run before the master is ready; if `MASTER_URL` is empty, registration and heartbeat are skipped.
- Live ingestion and live task execution call OpenAI and can spend API credits.
