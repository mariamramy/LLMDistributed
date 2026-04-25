# CSE354 Distributed Computing Project — Technical Plan
## Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Final Tech Stack](#2-final-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [Folder Structure](#4-folder-structure)
5. [The gRPC Contract](#5-the-grpc-contract)
6. [Layer 1 — Load Balancer](#6-layer-1--load-balancer)
7. [Layer 2 — gRPC Worker Server](#7-layer-2--grpc-worker-server)
8. [Layer 3 — RAG Module](#8-layer-3--rag-module)
9. [Layer 4 — OpenAI LLM Call](#9-layer-4--openai-llm-call)
10. [Layer 5 — Load Testing Client](#10-layer-5--load-testing-client)
11. [Layer 6 — Monitoring](#11-layer-6--monitoring)
12. [Docker Compose Setup](#12-docker-compose-setup)
13. [Fault Tolerance Implementation](#13-fault-tolerance-implementation)
14. [Implementation Phases](#14-implementation-phases)
15. [Testing and Demo Plan](#15-testing-and-demo-plan)
16. [Where AI Is Used](#16-where-ai-is-used)

---

## 1. Project Overview

This project builds a distributed system that handles 1000+ concurrent LLM requests using:

- **FastAPI** as the load balancer entry point
- **gRPC** for fast binary communication between the load balancer and worker nodes
- **OpenAI API** (`gpt-4o-mini`) as the LLM inference engine
- **ChromaDB + sentence-transformers** for Retrieval-Augmented Generation (RAG)
- **Locust** for realistic load simulation
- **Prometheus + Grafana** for observability
- **Docker Compose** to orchestrate all services locally

The key design decision is to **replace raw threading and direct function calls** with gRPC-based worker nodes. This means workers can be killed and restarted independently, and the load balancer detects failures and reroutes automatically — satisfying the fault tolerance requirement cleanly.

---

## 2. Final Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11 | Primary language |
| Load balancer | FastAPI + Uvicorn | Async HTTP server, routing logic |
| Worker communication | gRPC (`grpcio`) | Fast binary RPC between LB and workers |
| LLM inference | OpenAI API (`gpt-4o-mini`) | Answering user queries |
| RAG vector store | ChromaDB | Local vector database for context retrieval |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Embedding queries and documents |
| Load testing | Locust | Simulating 100–1000 concurrent users |
| Containerization | Docker + Docker Compose | Running all services locally |
| Monitoring | Prometheus + Grafana | Metrics, latency histograms, dashboards |
| Metrics library | `prometheus-fastapi-instrumentator` | Auto-expose FastAPI metrics |

---

## 3. System Architecture

### Request lifecycle

```
Locust Client (1000 users)
        |
        | POST /query
        v
FastAPI Load Balancer  ──────────────────────────────────
        |                                                |
        | gRPC call (picks worker via strategy)          | Prometheus
        v                                              metrics
gRPC Worker Node (one of 3)
        |
        |── RAG Module (ChromaDB query → top-3 chunks)
        |
        |── OpenAI API call (gpt-4o-mini)
        |      system prompt = RAG context
        |      user message  = original query
        |
        v
   Response (result + latency)
        |
        v
FastAPI returns JSON to client
```

### Load balancing strategies

The load balancer supports three strategies, selectable at startup via environment variable:

- **Round Robin** — cycles through workers sequentially
- **Least Connections** — picks the worker with the fewest active gRPC calls
- **Load-aware routing** — picks the worker with the lowest composite load score (active connections + recent latency)

### Fault tolerance mechanism

When a gRPC call fails or times out, the load balancer:

1. Catches the `grpc.RpcError` exception
2. Marks the worker as `healthy = False`
3. Retries the request on the next available healthy worker
4. A background `asyncio` task pings each worker's gRPC health endpoint every 5 seconds and restores `healthy = True` on recovery

---

## 4. Folder Structure

```
project/
├── docker-compose.yml
├── .env                          # OPENAI_API_KEY goes here
├── proto/
│   └── worker.proto              # gRPC service definition
├── lb/
│   ├── main.py                   # FastAPI load balancer
│   ├── strategies.py             # RoundRobin, LeastConn, LoadAware classes
│   └── worker_registry.py        # Worker health tracking
├── worker/
│   ├── server.py                 # gRPC worker server
│   ├── rag.py                    # ChromaDB retrieval
│   ├── llm.py                    # OpenAI API call
│   └── seed_db.py                # One-time script to seed ChromaDB
├── client/
│   └── locustfile.py             # Load test definition
├── monitoring/
│   ├── prometheus.yml            # Scrape config
│   └── grafana/
│       └── dashboard.json        # Pre-built dashboard
├── common/
│   └── models.py                 # Shared dataclasses
└── requirements.txt
```

---

## 5. The gRPC Contract

File: `proto/worker.proto`

This is the single source of truth for communication between the load balancer and every worker. Run the compiler to generate Python stubs before writing any other code.

```protobuf
syntax = "proto3";

package worker;

service WorkerService {
  rpc Process (QueryRequest) returns (QueryResponse);
  rpc HealthCheck (HealthRequest) returns (HealthResponse);
}

message QueryRequest {
  int32  id    = 1;
  string query = 2;
}

message QueryResponse {
  int32  id      = 1;
  string result  = 2;
  float  latency = 3;
  string worker_id = 4;
}

message HealthRequest {}

message HealthResponse {
  bool   healthy          = 1;
  int32  active_requests  = 2;
  float  avg_latency_ms   = 3;
}
```

**Generate Python stubs:**

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/worker.proto
```

This produces `worker_pb2.py` and `worker_pb2_grpc.py`. Both are imported by the load balancer and the worker servers.

---

## 6. Layer 1 — Load Balancer

File: `lb/main.py`

The load balancer is a FastAPI application. On startup it reads the worker hostnames from an environment variable, creates one gRPC channel per worker, and initialises the chosen strategy.

```python
# lb/main.py
import os, asyncio, time
import grpc
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

import worker_pb2, worker_pb2_grpc
from lb.strategies import RoundRobin, LeastConnections, LoadAware

app = FastAPI()
Instrumentator().instrument(app).expose(app)

ACTIVE_CONNS = Gauge("worker_active_connections", "Active gRPC calls", ["worker_id"])

class WorkerProxy:
    def __init__(self, host: str, port: int):
        self.id = f"{host}:{port}"
        self.healthy = True
        self.active_connections = 0
        self.avg_latency = 0.0
        channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self.stub = worker_pb2_grpc.WorkerServiceStub(channel)

workers: list[WorkerProxy] = []
strategy = None

@app.on_event("startup")
async def startup():
    global workers, strategy
    hosts = os.environ["WORKER_HOSTS"].split(",")  # e.g. "worker1:50051,worker2:50052"
    workers = [WorkerProxy(*h.split(":")) for h in hosts]

    mode = os.environ.get("LB_STRATEGY", "round_robin")
    strategy = {"round_robin": RoundRobin, "least_conn": LeastConnections,
                "load_aware": LoadAware}[mode](workers)

    asyncio.create_task(health_watchdog())

async def health_watchdog():
    """Ping every worker every 5 seconds; restore healthy flag on recovery."""
    while True:
        for w in workers:
            try:
                resp = await w.stub.HealthCheck(worker_pb2.HealthRequest(), timeout=2)
                w.healthy = resp.healthy
                w.avg_latency = resp.avg_latency_ms
            except grpc.RpcError:
                w.healthy = False
        await asyncio.sleep(5)

@app.post("/query")
async def query(payload: dict):
    request_id = payload.get("id", 0)
    user_query = payload.get("query", "")

    for _ in range(len(workers)):
        worker = strategy.pick()
        if not worker.healthy:
            continue
        try:
            worker.active_connections += 1
            ACTIVE_CONNS.labels(worker_id=worker.id).set(worker.active_connections)
            start = time.time()

            grpc_request = worker_pb2.QueryRequest(id=request_id, query=user_query)
            response = await worker.stub.Process(grpc_request, timeout=30)

            worker.avg_latency = (time.time() - start) * 1000
            return {"id": response.id, "result": response.result,
                    "latency": response.latency, "served_by": response.worker_id}
        except grpc.RpcError:
            worker.healthy = False  # mark dead, try next
        finally:
            worker.active_connections -= 1
            ACTIVE_CONNS.labels(worker_id=worker.id).set(worker.active_connections)

    return {"error": "All workers unavailable"}, 503
```

### Load balancing strategies

File: `lb/strategies.py`

```python
# lb/strategies.py
import threading

class RoundRobin:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0
        self._lock = threading.Lock()

    def pick(self):
        with self._lock:
            w = self.workers[self.index % len(self.workers)]
            self.index += 1
            return w

class LeastConnections:
    def __init__(self, workers):
        self.workers = workers

    def pick(self):
        return min(self.workers, key=lambda w: w.active_connections if w.healthy else float("inf"))

class LoadAware:
    """Composite score: active connections + normalised recent latency."""
    def __init__(self, workers):
        self.workers = workers

    def pick(self):
        def score(w):
            if not w.healthy:
                return float("inf")
            return w.active_connections + (w.avg_latency / 1000)
        return min(self.workers, key=score)
```

---

## 7. Layer 2 — gRPC Worker Server

File: `worker/server.py`

Each worker is a standalone Python process running a gRPC server. It implements two RPCs: `Process` (handles a query) and `HealthCheck` (responds to the watchdog).

```python
# worker/server.py
import asyncio, os, time, grpc
from grpc import aio
import worker_pb2, worker_pb2_grpc
from worker.rag import retrieve_context
from worker.llm import run_llm

WORKER_ID = os.environ.get("WORKER_ID", "worker-unknown")
PORT      = int(os.environ.get("GRPC_PORT", 50051))

active_requests = 0
recent_latencies: list[float] = []

class WorkerServicer(worker_pb2_grpc.WorkerServiceServicer):

    async def Process(self, request, context):
        global active_requests, recent_latencies
        active_requests += 1
        start = time.time()

        print(f"[{WORKER_ID}] Processing request {request.id}: {request.query[:50]}")

        # RAG step — retrieve relevant context from ChromaDB
        rag_context = await retrieve_context(request.query)

        # LLM step — call OpenAI API with context + query
        result = await run_llm(request.query, rag_context)

        latency = (time.time() - start) * 1000
        recent_latencies.append(latency)
        if len(recent_latencies) > 100:
            recent_latencies.pop(0)

        active_requests -= 1
        return worker_pb2.QueryResponse(
            id=request.id,
            result=result,
            latency=latency,
            worker_id=WORKER_ID
        )

    async def HealthCheck(self, request, context):
        avg = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
        return worker_pb2.HealthResponse(
            healthy=True,
            active_requests=active_requests,
            avg_latency_ms=avg
        )

async def serve():
    server = aio.server()
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    await server.start()
    print(f"[{WORKER_ID}] gRPC server listening on port {PORT}")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
```

---

## 8. Layer 3 — RAG Module

File: `worker/rag.py`

ChromaDB runs as a separate Docker container. The worker connects to it on startup, and retrieves the top-3 most relevant document chunks for each query using cosine similarity over sentence embeddings.

```python
# worker/rag.py
import chromadb
from sentence_transformers import SentenceTransformer

_client = chromadb.HttpClient(host="chromadb", port=8001)
_collection = _client.get_or_create_collection("knowledge_base")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

async def retrieve_context(query: str, n_results: int = 3) -> str:
    query_embedding = _embedder.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    chunks = results["documents"][0] if results["documents"] else []
    return "\n\n".join(chunks) if chunks else "No relevant context found."
```

### Seeding the knowledge base

File: `worker/seed_db.py`

Run this once before starting the system. It loads your documents and embeds them into ChromaDB.

```python
# worker/seed_db.py
import chromadb
from sentence_transformers import SentenceTransformer

documents = [
    "Distributed computing is a field of computer science that studies distributed systems...",
    "Load balancing refers to distributing workloads across multiple computing resources...",
    "Retrieval-Augmented Generation (RAG) combines retrieval systems with language models...",
    # Add 50+ paragraphs from any domain (Wikipedia articles, textbooks, etc.)
]

client = chromadb.HttpClient(host="localhost", port=8001)
collection = client.get_or_create_collection("knowledge_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedder.encode(documents).tolist()
ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(documents=documents, embeddings=embeddings, ids=ids)
print(f"Seeded {len(documents)} documents into ChromaDB.")
```

---

## 9. Layer 4 — OpenAI LLM Call

File: `worker/llm.py`

This is the **only place** the OpenAI API is called. Everything else in the system is pure distributed infrastructure. The function takes the user query and the RAG context, builds a structured prompt, and returns the model's response.

```python
# worker/llm.py
import os
from openai import AsyncOpenAI

_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_TEMPLATE = """You are a helpful assistant.
Use the following retrieved context to answer the user's question accurately.
If the context is not relevant, answer from your own knowledge.

Context:
{context}
"""

async def run_llm(query: str, context: str) -> str:
    response = await _client.chat.completions.create(
        model="gpt-4o-mini",     # cheap (~$0.15/1M input tokens), fast, accurate
        max_tokens=300,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
            {"role": "user",   "content": query}
        ]
    )
    return response.choices[0].message.content
```

### Why `gpt-4o-mini`

| Model | Input cost | Output cost | Latency |
|---|---|---|---|
| gpt-4o | $2.50 / 1M tokens | $10.00 / 1M tokens | ~1–3s |
| gpt-4o-mini | $0.15 / 1M tokens | $0.60 / 1M tokens | ~0.5–1s |
| gpt-3.5-turbo | $0.50 / 1M tokens | $1.50 / 1M tokens | ~0.5s |

`gpt-4o-mini` gives the best balance of cost, speed, and quality for a high-concurrency demo.

---

## 10. Layer 5 — Load Testing Client

File: `client/locustfile.py`

Locust simulates concurrent users. Each user sends a POST request to the load balancer with a random query. The `--headless` flag runs without a browser, and Locust outputs a live CSV of results.

```python
# client/locustfile.py
import random
from locust import HttpUser, task, between

SAMPLE_QUERIES = [
    "What is load balancing in distributed systems?",
    "Explain how RAG improves LLM responses.",
    "What is the difference between Round Robin and Least Connections?",
    "How does fault tolerance work in distributed computing?",
    "What are the benefits of GPU clusters for AI inference?",
    "Explain gRPC and why it is used for microservices.",
    "What is ChromaDB and how does vector search work?",
    "Describe the CAP theorem in distributed systems.",
    "What is horizontal scaling vs vertical scaling?",
    "How do heartbeat mechanisms detect node failures?",
]

class LLMUser(HttpUser):
    wait_time = between(0.5, 2.0)   # think time between requests

    @task
    def send_query(self):
        query = random.choice(SAMPLE_QUERIES)
        self.client.post("/query", json={
            "id": random.randint(1, 100000),
            "query": query
        }, timeout=60)
```

**Run commands:**

```bash
# Ramp to 1000 users at 50 users/second, run for 5 minutes
locust -f client/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  -u 1000 -r 50 \
  --run-time 5m \
  --csv results/load_test

# Open the live web UI (no --headless)
locust -f client/locustfile.py --host http://localhost:8000
```

---

## 11. Layer 6 — Monitoring

File: `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: load_balancer
    static_configs:
      - targets: ["lb:8000"]
```

The FastAPI load balancer auto-exposes these metrics at `/metrics` via `prometheus-fastapi-instrumentator`:

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Total requests by status code |
| `http_request_duration_seconds` | Histogram | Latency distribution (p50, p95, p99) |
| `worker_active_connections` | Gauge | Live connections per worker |

**Grafana dashboard panels to build:**

1. Requests per second (rate on `http_requests_total`)
2. p95 latency over time (histogram quantile)
3. Active connections per worker (stacked bar)
4. Error rate (5xx responses as % of total)

---

## 12. Docker Compose Setup

File: `docker-compose.yml`

```yaml
version: "3.9"

services:

  lb:
    build: ./lb
    ports:
      - "8000:8000"
    environment:
      - WORKER_HOSTS=worker1:50051,worker2:50052,worker3:50053
      - LB_STRATEGY=least_conn
    depends_on:
      - worker1
      - worker2
      - worker3

  worker1:
    build: ./worker
    environment:
      - WORKER_ID=worker-1
      - GRPC_PORT=50051
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - chromadb

  worker2:
    build: ./worker
    environment:
      - WORKER_ID=worker-2
      - GRPC_PORT=50052
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - chromadb

  worker3:
    build: ./worker
    environment:
      - WORKER_ID=worker-3
      - GRPC_PORT=50053
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - chromadb

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8001"
    volumes:
      - chroma_data:/chroma/chroma

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

volumes:
  chroma_data:
```

**Start everything:**

```bash
# Copy your API key into the env file
echo "OPENAI_API_KEY=sk-..." > .env

# Start all services
docker compose up --build

# Seed ChromaDB once (after containers are up)
python worker/seed_db.py

# Run load test
locust -f client/locustfile.py --host http://localhost:8000
```

---

## 13. Fault Tolerance Implementation

### Failure detection

The `health_watchdog` coroutine in the load balancer pings every worker's `HealthCheck` RPC every 5 seconds. If the call times out or raises a gRPC error, `worker.healthy` is set to `False`. When the worker recovers, the next successful ping restores it to `True`.

### Task reassignment on failure

When `worker.Process()` raises a `grpc.RpcError` (connection refused, deadline exceeded, etc.), the load balancer's retry loop moves to the next healthy worker. The original request is never dropped — it is retried up to `len(workers)` times before returning a 503.

```python
# Retry loop in lb/main.py (already shown above)
for _ in range(len(workers)):
    worker = strategy.pick()
    if not worker.healthy:
        continue
    try:
        response = await worker.stub.Process(grpc_request, timeout=30)
        return ...
    except grpc.RpcError:
        worker.healthy = False   # mark dead, try next
```

### Demo procedure for YouTube video

```bash
# Terminal 1: Start all services and run load test
docker compose up
locust -f client/locustfile.py --host http://localhost:8000 --headless -u 500 -r 20

# Terminal 2: Kill worker2 mid-test (simulates node failure)
docker stop project-worker2-1

# Observe in Grafana: active connections drop on worker2,
# increase on worker1 and worker3, error rate stays 0%

# Terminal 3: Restart worker2 (simulates recovery)
docker start project-worker2-1

# Observe: worker2 rejoins automatically after next health ping (~5s)
```

---

## 14. Implementation Phases

### Phase 1 — Architecture & Setup (week 1)

- [ ] Set up the folder structure
- [ ] Write and compile `worker.proto`, verify Python stubs generate correctly
- [ ] Write `docker-compose.yml` and confirm all containers start
- [ ] Implement the FastAPI stub (just returns a hardcoded response)
- [ ] Implement the gRPC worker stub (just returns `"ok"`)
- [ ] Confirm gRPC call works end-to-end from LB to worker

### Phase 2 — Core Implementation (week 2)

- [ ] Implement all three load balancing strategies in `lb/strategies.py`
- [ ] Implement `worker/server.py` with the real gRPC servicer
- [ ] Implement `worker/llm.py` with the OpenAI call
- [ ] Test with a single user manually via `curl http://localhost:8000/query`

### Phase 3 — RAG + Fault Tolerance (week 3)

- [ ] Implement `worker/rag.py` with ChromaDB retrieval
- [ ] Write and run `worker/seed_db.py` with 50+ documents
- [ ] Verify RAG context appears in prompts sent to OpenAI
- [ ] Implement `health_watchdog` in the load balancer
- [ ] Test fault tolerance: kill a worker, confirm requests reroute

### Phase 4 — Testing & Finalization (week 4–5)

- [ ] Run Locust at 100, 500, and 1000 users — record CSV results
- [ ] Set up Prometheus and Grafana, build the dashboard
- [ ] Record the YouTube demo video showing: normal operation → node failure → recovery
- [ ] Write the project report using the instructor's template
- [ ] Prepare the presentation

---

## 15. Testing and Evaluation

### Load testing matrix

| Users | Ramp rate | Expected p95 latency | Expected error rate |
|---|---|---|---|
| 100 | 10/s | < 2s | 0% |
| 500 | 20/s | < 4s | < 1% |
| 1000 | 50/s | < 8s | < 2% |

> Note: Latency is dominated by the OpenAI API response time (~0.5–1s). At very high concurrency, OpenAI rate limits may cause 429 errors — add a retry-with-backoff wrapper in `llm.py` if needed.

### Fault tolerance test cases

| Test | Action | Expected behaviour |
|---|---|---|
| Single worker failure | `docker stop worker2` | Requests reroute to worker1, worker3 within one retry |
| All workers fail | `docker stop worker1 worker2 worker3` | LB returns 503, no crash |
| Worker recovery | `docker start worker2` | Worker2 rejoins within 5–10 seconds |
| Slow worker | Add `time.sleep(5)` to one worker | Load-aware strategy stops routing to it |

### Performance metrics to report

- Throughput (requests/second at each load level)
- p50, p95, p99 latency
- Worker load distribution (should be roughly even with round robin)
- Error rate under failure conditions

---

## 16. Where AI Is Used

The OpenAI API is called in exactly **one place**: `worker/llm.py`, function `run_llm()`.

Everything else in the system is distributed infrastructure:

| Component | Is AI? | What it actually is |
|---|---|---|
| `lb/main.py` | No | Distributed load balancer |
| `lb/strategies.py` | No | Scheduling algorithms |
| `worker/server.py` | No | gRPC server / distributed worker |
| `worker/rag.py` | No (ML, not AI) | Vector similarity search |
| `worker/llm.py` | **Yes** | OpenAI API call |
| `client/locustfile.py` | No | Load testing framework |
| `monitoring/` | No | Observability infrastructure |

The RAG module uses `sentence-transformers` for embeddings — this is a machine learning model running locally, not a cloud AI API call. It converts text to vectors and measures cosine similarity. This is computation, not generation.

**The project's argument to the grader**: the OpenAI API replaces the "GPU worker doing LLM inference" from the spec. The distributed systems work — load balancing, fault detection, task reassignment, performance monitoring — is fully implemented and genuinely non-trivial. The AI is intentionally isolated so the infrastructure is clearly the focus.

---

## Requirements

File: `requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.30.0
grpcio==1.64.0
grpcio-tools==1.64.0
openai==1.35.0
chromadb==0.5.0
sentence-transformers==3.0.0
prometheus-fastapi-instrumentator==7.0.0
prometheus-client==0.20.0
locust==2.29.0
python-dotenv==1.0.1
```

Install: `pip install -r requirements.txt`