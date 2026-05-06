import json

import pytest

from llm.inference import LLMResult, SourceSnippet
from rag.retriever import RetrievalResult
from workers.gpu_workers import GPUWorker, WorkerConfig


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class FakeRetriever:
    def __init__(self):
        self.retrieval_count = 0

    async def is_ready(self):
        return True

    async def count(self):
        return 2

    async def retrieve(self, prompt):
        self.retrieval_count += 1
        return RetrievalResult(
            sources=[
                SourceSnippet(
                    text="replication context",
                    source_file="book.pdf",
                    page=4,
                    chunk_id="chunk-4",
                    score=0.9,
                )
            ]
        )


class FakeOllamaClient:
    def __init__(self, ready=True):
        self.ready = ready

    async def models_ready(self, generation_model, embedding_model):
        return self.ready


async def fake_llm_runner(prompt, sources, *, client, model, max_output_tokens):
    assert prompt == "Explain replication"
    assert len(sources) == 1
    return LLMResult(text="replication answer", model=model, usage={"output_tokens": 5})


def make_worker(retriever=None, ollama_client=None):
    config = WorkerConfig(
        worker_id="worker-test",
        host="127.0.0.1",
        port=9100,
        advertise_host="worker-test",
        master_url="",
        max_concurrency=4,
        heartbeat_interval_s=5,
        ollama_base_url="http://ollama:11434",
        generation_model="llama3.2:3b-instruct-q3_K_M",
        embedding_model="all-minilm",
        max_output_tokens=300,
    )
    return GPUWorker(
        config,
        retriever=retriever or FakeRetriever(),
        ollama_client=ollama_client or FakeOllamaClient(),
        llm_runner=fake_llm_runner,
    )


@pytest.mark.asyncio
async def test_health_reports_ready_worker():
    worker = make_worker()

    response = await worker.handle_health(FakeRequest({}))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["worker_id"] == "worker-test"
    assert payload["llm_provider"] == "ollama"
    assert payload["llm_ready"] is True
    assert payload["rag_ready"] is True
    assert payload["chunk_count"] == 2


@pytest.mark.asyncio
async def test_task_returns_answer_and_sources():
    worker = make_worker()

    response = await worker.handle_task(
        FakeRequest({"task_id": "task-1", "prompt": "Explain replication", "use_rag": True})
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["status"] == "completed"
    assert payload["result"] == "replication answer"
    assert payload["rag"]["sources"][0]["source_file"] == "book.pdf"
    assert worker.metrics.completed_tasks == 1


@pytest.mark.asyncio
async def test_task_rejects_unready_ollama():
    config = WorkerConfig(
        worker_id="worker-test",
        host="127.0.0.1",
        port=9100,
        advertise_host="worker-test",
        master_url="",
        max_concurrency=4,
        heartbeat_interval_s=5,
        ollama_base_url="http://ollama:11434",
        generation_model="llama3.2:3b-instruct-q3_K_M",
        embedding_model="all-minilm",
        max_output_tokens=300,
    )
    worker = GPUWorker(
        config,
        retriever=FakeRetriever(),
        ollama_client=FakeOllamaClient(ready=False),
    )

    response = await worker.handle_task(
        FakeRequest({"task_id": "task-1", "prompt": "Explain replication"})
    )
    payload = json.loads(response.text)

    assert response.status == 503
    assert payload["error"]["code"] == "ollama_not_ready"


@pytest.mark.asyncio
async def test_metrics_exposes_ollama_errors_counter():
    worker = make_worker()

    response = await worker.handle_metrics(FakeRequest({}))
    payload = json.loads(response.text)

    assert response.status == 200
    assert "ollama_errors" in payload
    assert "openai_errors" not in payload
