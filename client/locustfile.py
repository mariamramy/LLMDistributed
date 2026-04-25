import random

from locust import HttpUser, between, task


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
    wait_time = between(0.5, 2.0)

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
