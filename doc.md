# LLMDistributed — Project Documentation

---

## Teammate Quick Start

> Everything you need to run the full project from scratch. Read this first.

### What you need before starting
- **Docker Desktop** installed and running (with GPU support enabled)
- **NVIDIA GPU** with CUDA support and at least 6GB VRAM
- **Python 3.11+** installed locally
- **The textbook PDF** — get it from Noha and place it in the `pdfs/` folder (create the folder if it doesn't exist)

### Step 1 — Clone and set up
```bash
git clone <repo-url>
cd LLMDistributed
```

Create the `pdfs/` folder and drop the textbook PDF inside it:
```bash
mkdir pdfs
# copy the PDF file into pdfs/
```

Copy the environment file:
```bash
cp .env.example .env
```
You do not need to edit `.env` — the defaults work out of the box.

### Step 2 — Start everything
```powershell
docker compose up -d
```

This will:
1. Download Ollama and ChromaDB Docker images (~1GB first time)
2. Pull `llama3.2:3b-instruct-q3_K_M` (~1.7GB) and `all-minilm` (~46MB) — **first run only, takes 5-10 minutes**
3. Ingest the PDF into ChromaDB — **takes 3-5 minutes**
4. Start all 3 workers

**Be patient on the first run.** Subsequent runs start in under 60 seconds.

### Step 3 — Verify everything is running
```powershell
docker compose ps
```
All services should show `Up`. Then check workers are registered:
```powershell
curl http://localhost:9000/stats
```
You should see 3 workers with `"healthy": true`. If you see 5 workers, the Thundercompute nodes are also connected.

### Step 4 — Send a test request
```powershell
curl -X POST http://localhost:8080/request -H "Content-Type: application/json" -d "{\"prompt\": \"What is the CAP theorem?\", \"use_rag\": false}"
```
You should get back a JSON response with `"status": "completed"` and an answer from the LLM.

### Step 5 — Run the 1000-request load test
```powershell
python tests/integration_load_test.py --requests 1000 --concurrency 20 --timeout 300
```
This runs the full pipeline and prints a report at the end showing throughput, latency, success rate, and how many requests each worker handled.

### Step 6 — Shut down when done
```powershell
docker compose down
```

---

### Common issues

**Workers not starting** — `rag_ingest` is still running (embedding the PDF). Wait for it to finish: `docker compose logs rag_ingest`

**Port already in use** — something else is using 8080, 9000, 8000, or 11434. Stop it or change the port in `docker-compose.yml`

**Out of memory / GPU error** — close other GPU-heavy applications. The model needs ~2GB VRAM minimum.

**worker-4 / worker-5 unhealthy** — Thundercompute instance is offline. Ignore it — the system runs fine on 3 local workers only.

---

## Overview

A fully distributed LLM inference system built for CSE354. Client requests flow through a load balancer to a master scheduler, which distributes tasks across multiple GPU workers. Each worker performs RAG (Retrieval-Augmented Generation) using a local vector database and generates answers using a local LLM via Ollama.

The system supports hybrid deployments — mixing local GPU workers with remote cloud GPU workers (Thundercompute) in the same cluster.

---

## Architecture

```
Client
  │
  ▼
Load Balancer (port 8080)
  │   Round-robin / Least-connections / Load-aware routing
  ▼
Master Scheduler (port 9000)
  │   Priority queue, async dispatch, fault tolerance, retries
  ├──▶ Worker-1 (port 9101) ──▶ Local Ollama (port 11434) ──▶ GPU
  ├──▶ Worker-2 (port 9102) ──▶ Local Ollama (port 11434) ──▶ GPU
  ├──▶ Worker-3 (port 9103) ──▶ Local Ollama (port 11434) ──▶ GPU
  ├──▶ Worker-4 (port 9104) ──▶ Thundercompute Ollama (HTTPS) ──▶ Cloud GPU
  └──▶ Worker-5 (port 9105) ──▶ Thundercompute Ollama (HTTPS) ──▶ Cloud GPU
            │
            ▼
        ChromaDB (port 8000)
        Vector store for RAG
```

### Components

| Component | Port | Description |
|---|---|---|
| Load Balancer | 8080 | Receives client requests, forwards to master |
| Master Scheduler | 9000 | Priority queue, dispatches tasks to workers |
| Worker 1-3 | 9101-9103 | Local GPU workers (RAG + LLM inference) |
| Worker 4-5 | 9104-9105 | Cloud GPU workers (Thundercompute) |
| ChromaDB | 8000 | Vector database for RAG retrieval |
| Ollama | 11434 | Local LLM inference server |

---

## Full Request Flow

1. **Client** sends `POST /request` to Load Balancer with a prompt
2. **Load Balancer** selects master using its strategy (round-robin by default) and forwards
3. **Master Scheduler** wraps request in a `Task`, assigns a UUID, puts it in a `PriorityQueue`, creates an asyncio `Future`
4. **Dispatch loop** picks the best available worker (lowest load + active tasks), increments its counter, fires off `asyncio.create_task()`
5. **Worker** receives `POST /task`:
   - Checks readiness of Ollama and ChromaDB (cached for 30s)
   - If `use_rag=True`: embeds the prompt via Ollama, queries ChromaDB for top-K chunks
   - Builds prompt with retrieved context
   - Calls Ollama `/api/chat` for LLM generation
   - Returns JSON with answer, latency, worker ID, RAG sources
6. **Master** resolves the Future, returns result to LB
7. **LB** returns result to client

---

## Prerequisites

- Docker Desktop with GPU support enabled
- NVIDIA GPU with CUDA support
- Python 3.11+ (for running the load test client)
- At least 6GB VRAM (for `llama3.2:3b-instruct-q3_K_M`)
- PDF textbooks placed in the `pdfs/` directory

---

## Setup and Running

### 1. Clone and configure

```bash
git clone <repo-url>
cd LLMDistributed
cp .env.example .env
```

Edit `.env` as needed (see Configuration section).

### 2. Add PDF textbooks

Place your textbook PDFs in the `pdfs/` directory. The RAG pipeline will automatically chunk, embed, and index them at startup.

### 3. Start the full stack

```bash
docker compose up --build
```

The first run will:
- Pull the Ollama Docker image (~1GB)
- Download `llama3.2:3b-instruct-q3_K_M` (~1.7GB) and `all-minilm` (~46MB)
- Ingest PDFs into ChromaDB (takes a few minutes)
- Start all workers

Subsequent runs are much faster — models are cached in Docker volumes.

### 4. Verify everything is healthy

```bash
# Load Balancer
curl http://localhost:8080/health

# Master Scheduler
curl http://localhost:9000/health
curl http://localhost:9000/stats

# Individual workers
curl http://localhost:9101/health
curl http://localhost:9102/health
curl http://localhost:9103/health

# ChromaDB
curl http://localhost:8000/api/v2/heartbeat

# Ollama
curl http://localhost:11434/api/tags
```

### 5. Send a test request

```bash
curl -X POST http://localhost:8080/request \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the CAP theorem?",
    "use_rag": true
  }'
```

---

## Running the Load Test

```bash
python tests/integration_load_test.py --requests 1000 --concurrency 20 --timeout 300
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--requests` | 1000 | Total number of requests to send |
| `--concurrency` | 10 | Max simultaneous in-flight requests |
| `--timeout` | 300 | Per-request timeout in seconds |

### Output format

Progress line (printed every 50 requests):
```
[ 500/1000]  50.0% | OK=498 FAIL=2 | 0.86 req/s | 577s elapsed
```

- `OK` / `FAIL` — successful vs failed requests
- `req/s` — current throughput (completed / elapsed time)

Final report includes: total time, throughput, success rate, latency percentiles (P50/P95/P99), and requests per worker.

---

## Configuration

All settings live in `.env`. Key variables:

### LLM Settings

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama server URL (override per worker for hybrid) |
| `OLLAMA_GENERATION_MODEL` | `llama3.2:3b-instruct-q3_K_M` | Generation model |
| `OLLAMA_EMBEDDING_MODEL` | `all-minilm` | Embedding model for RAG |
| `OLLAMA_MAX_OUTPUT_TOKENS` | `100` | Max tokens per response |
| `OLLAMA_EMBED_MAX_CHARS` | `200` | Max characters sent for embedding |
| `OLLAMA_SSL_VERIFY` | `true` | Set to `false` for self-signed HTTPS (Thundercompute) |

### Worker Settings

| Variable | Default | Description |
|---|---|---|
| `WORKER_MAX_CONCURRENCY` | `2` | Semaphore limit per worker |
| `HEARTBEAT_INTERVAL_S` | `5` | Seconds between heartbeats to master |
| `MASTER_URL` | `http://scheduler:9000` | Master scheduler URL |

### RAG Settings

| Variable | Default | Description |
|---|---|---|
| `RAG_TOP_K` | `1` | Number of chunks retrieved per query |
| `RAG_CHUNK_SIZE` | `500` | Characters per chunk during ingestion |
| `RAG_CHUNK_OVERLAP` | `75` | Overlap between chunks |
| `CHROMA_COLLECTION` | `distributed_systems_textbook_ollama_all_minilm` | Collection name |

### Timeout Settings

| Variable | Default | Description |
|---|---|---|
| `LB_FORWARD_TIMEOUT_S` | `250` | LB timeout waiting for master response |
| Master `forward_timeout_s` | `250` | Master timeout waiting for worker response |

---

## API Reference

### Load Balancer

| Endpoint | Method | Description |
|---|---|---|
| `/request` | POST | Submit a request (main entry point) |
| `/health` | GET | LB health and node stats |
| `/stats` | GET | Request counts and failure stats |

**Request body:**
```json
{
  "request_id": "optional-client-id",
  "prompt": "Your question here",
  "use_rag": true,
  "priority": 5
}
```

### Master Scheduler

| Endpoint | Method | Description |
|---|---|---|
| `/request` | POST | Receive from LB, queue task |
| `/register` | POST | Worker registration |
| `/heartbeat` | POST | Worker heartbeat |
| `/health` | GET | Queue size and worker count |
| `/stats` | GET | Full task and worker statistics |

### Workers

| Endpoint | Method | Description |
|---|---|---|
| `/task` | POST | Execute a task (called by master) |
| `/health` | GET | Worker readiness |
| `/metrics` | GET | Detailed performance metrics |

---

## Fault Tolerance

### Worker failures
If a worker returns a non-200 response or an exception occurs during dispatch, the master re-queues the task and retries on a different worker up to `MAX_TASK_RETRIES = 3` times.

### Worker timeouts
The `heartbeat_check_loop` runs every 5 seconds. Workers that miss heartbeats for more than `WORKER_TIMEOUT_S = 15` seconds are marked unhealthy. Any in-flight tasks on dead workers are re-queued and reassigned.

### Client retries
The load test client retries failed requests up to 2 times with a 2-second delay before counting them as failures.

### Demonstrated fault tolerance
During testing, Thundercompute workers failed with OOM errors (`OLLAMA_NUM_PARALLEL=8` exceeded available VRAM). The master scheduler automatically retried all affected tasks on healthy local workers. The client observed **zero failures** — demonstrating transparent fault recovery under partial node failure.

---

## Load Balancing Strategies

Three strategies available, set via `--strategy` flag or `STRATEGIES` config:

| Strategy | Description |
|---|---|
| `round_robin` | Alternates between healthy nodes equally |
| `least_connections` | Routes to node with fewest active connections |
| `load_aware` | Routes to node with lowest `(load, active_connections)` score |

Within the master scheduler, `get_best_worker()` always uses load-aware selection: `min(healthy_workers, key=lambda w: (w.load, w.active_tasks))`.

---

## Hybrid Cloud Setup (Thundercompute)

To add remote GPU workers:

### 1. Start Ollama on the remote instance

```bash
OLLAMA_HOST=0.0.0.0 OLLAMA_NUM_PARALLEL=3 nohup ollama serve > ollama.log 2>&1 &
ollama pull llama3.2:3b-instruct-q3_K_M
ollama pull all-minilm
```

**Important:** Set `OLLAMA_NUM_PARALLEL` based on available VRAM:
- ~9GB per parallel slot for this model
- A6000 (48GB): max 3-4 parallel slots safely (`OLLAMA_NUM_PARALLEL=3`)
- Do not exceed available VRAM or Ollama will return 500 errors

### 2. Add workers to docker-compose.yml

```yaml
worker-4:
  <<: *app
  command: ["python", "-m", "workers.gpu_workers"]
  ports:
    - "9104:9100"
  environment:
    <<: *app-env
    WORKER_ID: worker-4
    WORKER_PORT: "9100"
    WORKER_ADVERTISE_HOST: worker-4
    OLLAMA_BASE_URL: https://your-instance-url.thundercompute.net
    OLLAMA_SSL_VERIFY: "false"
    WORKER_MAX_CONCURRENCY: "4"
  depends_on:
    chromadb:
      condition: service_healthy
    rag_ingest:
      condition: service_completed_successfully
```

### 3. Start the new workers

```bash
docker compose up -d worker-4 worker-5
```

---

## Performance Results

All tests: 1000 requests, `llama3.2:3b-instruct-q3_K_M`, RTX 4050 laptop (6GB VRAM).

| Run | Setup | Concurrency | Success | Throughput | Total Time |
|---|---|---|---|---|---|
| 1 | 3 local workers | 10 | 100% | 0.54 req/s | 31.0 min |
| 2 | 3 local, optimized config | 10 | 99.1% | 0.55 req/s | 30.6 min |
| 3 | 3 local + 2 TC (A6000) | 10 | 99.5% | 0.75 req/s | 22.3 min |
| 4 | 3 local + 2 TC (TC broken) | 20 | 100% | 0.84 req/s | 19.8 min |
| 5 | 3 local + 2 TC (TC fixed) | 20 | 100% | 0.77 req/s | 21.7 min |
| 6 | 3 local only | 20 | 100% | 0.60 req/s | ~28 min est. |

### Key observations

- **Hybrid outperforms local-only:** Adding 2 Thundercompute A6000 workers improved throughput by ~28% (0.60 → 0.77 req/s) with all 3 local workers still active
- **Fault tolerance proven:** Run 4 showed 100% success even when TC workers were silently failing — master retried all failed tasks on local workers
- **Model size matters more than GPU:** The 3B quantized model runs at similar speed on RTX 4050 and A6000. The A6000 advantage only shows with larger models (>13B) that don't fit in 6GB VRAM
- **Throughput is Ollama-bound:** The bottleneck is LLM inference time (~5-30s per request), not the Python workers or network

---

## Optimizations Applied

### Bug fixes

| Fix | File | Impact |
|---|---|---|
| Integration test was hitting worker directly instead of LB | `tests/integration_load_test.py` | Test now measures the full pipeline |
| Dispatch loop race condition: same worker selected multiple times before active_tasks updated | `master/scheduler.py` | Even load distribution across all workers (334/332/334) |

### Performance optimizations

| Optimization | File | Impact |
|---|---|---|
| Readiness check caching (30s TTL) | `workers/gpu_workers.py` | Eliminates Ollama + ChromaDB HTTP calls on every task |
| SSL bypass for self-signed HTTPS | `llm/inference.py` | Enables Thundercompute HTTPS connections |
| `OLLAMA_FLASH_ATTENTION=1` | `docker-compose.yml` | Reduces VRAM usage, speeds up attention |
| `WORKER_MAX_CONCURRENCY` 1→2 | `.env` | Workers handle 2 concurrent tasks each |
| Master + LB timeouts 120s→250s | `master/scheduler.py`, `docker-compose.yml` | Eliminates timeout failures under high concurrency |
| Client retry logic (2 retries) | `tests/integration_load_test.py` | Recovers transient failures automatically |

---

## Troubleshooting

### Workers not appearing in master stats after restart

Workers only register at startup. After a master restart, restart all workers:
```bash
docker compose restart worker-1 worker-2 worker-3 worker-4 worker-5
```

### Thundercompute OOM errors

If workers 4/5 return 502 with `"model requires more system memory"`, reduce `OLLAMA_NUM_PARALLEL` on the Thundercompute instance. Each parallel slot needs ~9GB for this model.

### RAG ingestion fails on restart

`rag_ingest` re-runs when workers are force-recreated. Wait for it to complete (check `docker compose logs rag_ingest`) before starting the load test.

### GPU thermal throttling

Sustained load on a laptop GPU causes thermal throttling above ~85°C, reducing throughput over time. Ensure airflow under the laptop and monitor with `nvidia-smi`.

---

## Project Structure

```
LLMDistributed/
├── client/
│   └── load_generator.py       # Async load generator with stats
├── common/
│   └── models.py               # Shared Request/Response dataclasses
├── lb/
│   ├── load_balancer.py        # aiohttp LB server
│   ├── health_monitor.py       # Background node health checks
│   ├── node.py                 # Node dataclass
│   └── strategies.py           # RoundRobin, LeastConnections, LoadAware
├── master/
│   ├── scheduler.py            # Master scheduler with priority queue
│   ├── work_registry.py        # WorkerRegistry and TaskStore
│   └── models.py               # Task, WorkerInfo, TaskStatus
├── workers/
│   └── gpu_workers.py          # GPU worker: RAG + LLM + heartbeat
├── rag/
│   ├── ingest.py               # PDF → ChromaDB ingestion pipeline
│   └── retriever.py            # ChromaDB query with Ollama embeddings
├── llm/
│   └── inference.py            # OllamaClient wrapper (chat + embed)
├── tests/
│   ├── integration_load_test.py  # Full pipeline load test (1000 requests)
│   ├── integration_parallel.py   # Direct worker parallel test
│   ├── test_worker.py
│   ├── test_retriever.py
│   ├── test_ingest.py
│   └── test_llm_inference.py
├── postman/                    # Postman collection for manual testing
├── pdfs/                       # Textbook PDFs (git-ignored)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```
