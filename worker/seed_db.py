from __future__ import annotations

import os

import chromadb

from worker.embeddings import HashEmbeddingFunction


DOCUMENTS = [
    "Distributed computing studies systems whose components run on networked computers and coordinate by passing messages.",
    "A distributed system can improve scalability by splitting work across multiple machines instead of relying on one large server.",
    "Load balancing distributes incoming requests across worker nodes to improve throughput and avoid overloaded machines.",
    "Round robin load balancing cycles through workers in a fixed order and is simple to implement.",
    "Least connections load balancing sends new work to the node with the fewest active requests.",
    "Load-aware routing combines signals such as active requests, latency, CPU use, or queue depth to choose a worker.",
    "Fault tolerance is the ability of a system to continue operating when one or more components fail.",
    "A heartbeat is a periodic health signal used to detect whether a service is still reachable.",
    "A watchdog task can mark unreachable workers as unhealthy and restore them when health checks succeed again.",
    "Task reassignment helps preserve availability by retrying failed work on another healthy node.",
    "gRPC is a high-performance RPC framework that uses Protocol Buffers for compact binary serialization.",
    "Protocol Buffers define service contracts and message schemas that can be compiled into client and server code.",
    "FastAPI is an asynchronous Python web framework suitable for exposing HTTP APIs with automatic validation.",
    "Uvicorn runs ASGI applications and can serve FastAPI apps efficiently in development and production demos.",
    "ChromaDB is a vector database used to store embeddings and retrieve semantically similar documents.",
    "Sentence transformers convert text into dense vectors that capture semantic meaning.",
    "Retrieval-Augmented Generation retrieves relevant documents before generating an answer with a language model.",
    "RAG can reduce hallucinations by grounding model responses in retrieved context.",
    "Cosine similarity compares vector directions and is commonly used for semantic search.",
    "Embedding models map related phrases to nearby points in vector space.",
    "OpenAI API calls can provide LLM inference while the surrounding system demonstrates distributed infrastructure.",
    "The gpt-4o-mini model is useful for demos because it is fast and cost efficient.",
    "Prometheus scrapes metrics from instrumented services and stores them as time-series data.",
    "Grafana visualizes time-series metrics using dashboards, panels, and queries.",
    "Latency histograms show the distribution of request durations and help estimate p95 and p99 performance.",
    "A p95 latency value means that ninety-five percent of requests completed at or below that time.",
    "Locust simulates concurrent users by running Python user classes that send HTTP requests.",
    "A load test ramp rate controls how quickly virtual users are added to the test.",
    "Horizontal scaling adds more machines or containers, while vertical scaling gives a machine more resources.",
    "Backpressure prevents a system from accepting unlimited work when downstream services are overloaded.",
    "Timeouts prevent clients from waiting forever when a service fails or becomes slow.",
    "Retries can improve reliability, but uncontrolled retries can amplify load during failures.",
    "A retry budget limits the number of attempts and protects the system from retry storms.",
    "Circuit breakers stop sending traffic to unhealthy services for a recovery interval.",
    "Observability combines metrics, logs, and traces to explain system behavior.",
    "A container packages application code and dependencies so it can run consistently across environments.",
    "Docker Compose starts multiple related containers and creates a shared network for service discovery.",
    "Service discovery lets containers refer to each other by service name, such as worker1 or chromadb.",
    "A health endpoint exposes whether a service is ready to handle traffic.",
    "A rolling average smooths latency measurements over recent requests.",
    "GPU clusters can accelerate AI inference by parallelizing computation across specialized hardware.",
    "Batching can improve inference throughput by processing several requests together.",
    "Queue length is a useful load signal because it reflects work waiting to be processed.",
    "CAP theorem states that distributed data systems trade off consistency, availability, and partition tolerance.",
    "Consistency means all clients observe the same data after updates complete.",
    "Availability means every request receives a non-error response from a live node.",
    "Partition tolerance means the system continues despite network splits or dropped messages.",
    "Idempotent operations can be safely retried without changing the final result unexpectedly.",
    "A worker process can fail independently from the load balancer in a microservice architecture.",
    "The load balancer should isolate worker failures so one bad node does not crash the entry point.",
    "Structured metrics make it easier to compare load balancing algorithms under the same traffic pattern.",
    "A demo should show normal routing, worker failure, automatic rerouting, and worker recovery.",
    "The project isolates AI generation in one module so the distributed systems behavior remains easy to evaluate.",
]


def main() -> None:
    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "hash")
    embedder = None
    if embedding_backend == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(
                os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
        except ImportError:
            embedder = None

    client = chromadb.HttpClient(
        host=os.environ.get("CHROMA_SEED_HOST", "localhost"),
        port=int(os.environ.get("CHROMA_SEED_PORT", "8001")),
    )
    collection = client.get_or_create_collection(
        os.environ.get("CHROMA_COLLECTION", "knowledge_base"),
        metadata={"hnsw:space": "cosine"},
        embedding_function=None if embedder else HashEmbeddingFunction(),
    )
    ids = [f"doc_{index}" for index in range(len(DOCUMENTS))]
    if embedder:
        embeddings = embedder.encode(DOCUMENTS).tolist()
        collection.upsert(documents=DOCUMENTS, embeddings=embeddings, ids=ids)
    else:
        collection.upsert(documents=DOCUMENTS, ids=ids)
    print(f"Seeded {len(DOCUMENTS)} documents into ChromaDB.")


if __name__ == "__main__":
    main()
