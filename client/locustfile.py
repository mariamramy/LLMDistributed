import random
import os

from locust import HttpUser, between, task


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


WAIT_MIN_SECONDS = _float_env("LOCUST_WAIT_MIN", 0.5)
WAIT_MAX_SECONDS = _float_env("LOCUST_WAIT_MAX", 2.0)
if WAIT_MIN_SECONDS > WAIT_MAX_SECONDS:
    WAIT_MIN_SECONDS, WAIT_MAX_SECONDS = WAIT_MAX_SECONDS, WAIT_MIN_SECONDS


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
    wait_time = between(WAIT_MIN_SECONDS, WAIT_MAX_SECONDS)

    @task
    def send_query(self):
        self.client.post(
            "/query",
            json={
                "id": random.randint(1, 100000),
                "query": random.choice(SAMPLE_QUERIES),
            },
            timeout=60,
        )
