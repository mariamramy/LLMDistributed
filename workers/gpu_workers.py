import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

from llm.inference import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OLLAMA_BASE_URL,
    LLMResult,
    SourceSnippet,
    create_ollama_client,
    generate_answer,
)
from rag.retriever import DEFAULT_EMBEDDING_MODEL, ChromaRAGRetriever


logging.basicConfig(level=logging.INFO, format="%(asctime)s [WORKER] %(levelname)s %(message)s")
log = logging.getLogger("gpu_worker")


@dataclass
class WorkerConfig:
    worker_id: str
    host: str
    port: int
    advertise_host: str
    master_url: str
    max_concurrency: int
    heartbeat_interval_s: float
    ollama_base_url: str
    generation_model: str
    embedding_model: str
    max_output_tokens: int

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        worker_id = os.getenv("WORKER_ID", os.getenv("HOSTNAME", "worker-1"))
        return cls(
            worker_id=worker_id,
            host=os.getenv("WORKER_HOST", "0.0.0.0"),
            port=int(os.getenv("WORKER_PORT", "9100")),
            advertise_host=os.getenv("WORKER_ADVERTISE_HOST", worker_id),
            master_url=os.getenv("MASTER_URL", "").rstrip("/"),
            max_concurrency=int(os.getenv("WORKER_MAX_CONCURRENCY", "1")),
            heartbeat_interval_s=float(os.getenv("HEARTBEAT_INTERVAL_S", "5")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            generation_model=os.getenv("OLLAMA_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            max_output_tokens=int(
                os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))
            ),
        )


@dataclass
class WorkerMetrics:
    active_tasks: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_latency_ms: float = 0.0
    retrieval_count: int = 0
    ollama_errors: int = 0
    chroma_errors: int = 0

    def avg_latency_ms(self) -> float:
        completed = max(1, self.completed_tasks)
        return self.total_latency_ms / completed


LLMRunner = Callable[..., Awaitable[LLMResult]]


class GPUWorker:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        retriever: Optional[ChromaRAGRetriever] = None,
        ollama_client: Optional[Any] = None,
        llm_runner: LLMRunner = generate_answer,
    ):
        self.config = config
        self.metrics = WorkerMetrics()
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._ollama_client = ollama_client
        self._retriever = retriever
        self._llm_runner = llm_runner
        # Cache readiness results for READY_CACHE_TTL_S seconds to avoid
        # hitting Ollama/ChromaDB on every incoming task request.
        self._llm_ready_cache: Optional[Tuple[bool, float]] = None
        self._chroma_ready_cache: Optional[Tuple[bool, float]] = None
        self._ready_cache_ttl_s: float = 30.0

    @property
    def load(self) -> float:
        return min(1.0, self.metrics.active_tasks / max(1, self.config.max_concurrency))

    @property
    def ollama_client(self) -> Any:
        if self._ollama_client is None:
            self._ollama_client = create_ollama_client(self.config.ollama_base_url)
        return self._ollama_client

    @property
    def retriever(self) -> ChromaRAGRetriever:
        if self._retriever is None:
            self._retriever = ChromaRAGRetriever(
                ollama_client=self.ollama_client,
                embedding_model=self.config.embedding_model,
            )
        return self._retriever

    async def llm_ready(self) -> bool:
        now = time.monotonic()
        if self._llm_ready_cache is not None:
            result, ts = self._llm_ready_cache
            if now - ts < self._ready_cache_ttl_s:
                return result
        try:
            result = await self.ollama_client.models_ready(
                self.config.generation_model,
                self.config.embedding_model,
            )
        except Exception:
            self.metrics.ollama_errors += 1
            result = False
        self._llm_ready_cache = (result, now)
        return result

    async def chroma_ready(self) -> bool:
        now = time.monotonic()
        if self._chroma_ready_cache is not None:
            result, ts = self._chroma_ready_cache
            if now - ts < self._ready_cache_ttl_s:
                return result
        try:
            result = await self.retriever.is_ready()
        except Exception:
            self.metrics.chroma_errors += 1
            result = False
        self._chroma_ready_cache = (result, now)
        return result

    async def rag_ready(self) -> bool:
        llm_ready, chroma_ready = await asyncio.gather(self.llm_ready(), self.chroma_ready())
        return llm_ready and chroma_ready

    async def chunk_count(self) -> int:
        try:
            return await self.retriever.count()
        except Exception:
            self.metrics.chroma_errors += 1
            return 0

    async def readiness(self) -> Dict[str, Any]:
        llm_ready, chroma_ready = await asyncio.gather(self.llm_ready(), self.chroma_ready())
        rag_ready = llm_ready and chroma_ready
        chunk_count = await self.chunk_count() if chroma_ready else 0
        return {
            "worker_id": self.config.worker_id,
            "status": "ok" if rag_ready else "unready",
            "llm_provider": "ollama",
            "llm_ready": llm_ready,
            "rag_ready": rag_ready,
            "chroma_ready": chroma_ready,
            "chunk_count": chunk_count,
            "active_tasks": self.metrics.active_tasks,
            "total_tasks": self.metrics.total_tasks,
            "load": self.load,
            "model": self.config.generation_model,
            "embedding_model": self.config.embedding_model,
        }

    async def handle_health(self, request: web.Request) -> web.Response:
        readiness = await self.readiness()
        status = 200 if readiness["status"] == "ok" else 503
        return web.json_response(readiness, status=status)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        llm_ready, chroma_ready = await asyncio.gather(self.llm_ready(), self.chroma_ready())
        rag_ready = llm_ready and chroma_ready
        chunk_count = await self.chunk_count() if chroma_ready else 0
        self.metrics.retrieval_count = getattr(self._retriever, "retrieval_count", 0)
        payload = {
            "worker_id": self.config.worker_id,
            "active_tasks": self.metrics.active_tasks,
            "max_concurrency": self.config.max_concurrency,
            "load": self.load,
            "total_tasks": self.metrics.total_tasks,
            "completed_tasks": self.metrics.completed_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "avg_latency_ms": round(self.metrics.avg_latency_ms(), 2),
            "llm_provider": "ollama",
            "llm_ready": llm_ready,
            "rag_ready": rag_ready,
            "chroma_ready": chroma_ready,
            "chunk_count": chunk_count,
            "retrieval_count": self.metrics.retrieval_count,
            "ollama_errors": self.metrics.ollama_errors,
            "chroma_errors": self.metrics.chroma_errors,
        }
        return web.json_response(payload)

    async def handle_task(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"status": "failed", "error": {"code": "invalid_json"}}, status=400)

        task_id = str(payload.get("task_id", "")).strip()
        request_id = payload.get("request_id")
        prompt = str(payload.get("prompt") or payload.get("query") or "").strip()
        use_rag = bool(payload.get("use_rag", True))

        if not task_id:
            return self._task_error(task_id, "missing_task_id", "task_id is required", status=400)
        if not prompt:
            return self._task_error(task_id, "missing_prompt", "prompt is required", status=400)
        if not await self.llm_ready():
            return self._task_error(
                task_id,
                "ollama_not_ready",
                "Ollama is not reachable or required models are not installed",
                status=503,
            )
        if use_rag and not await self.chroma_ready():
            return self._task_error(
                task_id,
                "rag_not_ready",
                "ChromaDB is not reachable or the textbook collection has no chunks",
                status=503,
            )

        async with self._semaphore:
            self.metrics.active_tasks += 1
            self.metrics.total_tasks += 1
            started = time.perf_counter()
            try:
                sources: List[SourceSnippet] = []
                if use_rag:
                    retrieval = await self.retriever.retrieve(prompt)
                    sources = retrieval.sources

                llm_result = await self._llm_runner(
                    prompt,
                    sources,
                    client=self.ollama_client,
                    model=self.config.generation_model,
                    max_output_tokens=self.config.max_output_tokens,
                )

                latency_ms = (time.perf_counter() - started) * 1000
                self.metrics.completed_tasks += 1
                self.metrics.total_latency_ms += latency_ms
                return web.json_response(
                    {
                        "task_id": task_id,
                        "request_id": request_id,
                        "status": "completed",
                        "worker_id": self.config.worker_id,
                        "result": llm_result.text,
                        "latency_ms": round(latency_ms, 2),
                        "rag": {
                            "used": use_rag,
                            "sources": [source_to_response(source) for source in sources],
                        },
                        "llm": {
                            "model": llm_result.model,
                            "usage": llm_result.usage,
                        },
                    }
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                self.metrics.failed_tasks += 1
                if "chroma" in exc.__class__.__name__.lower():
                    self.metrics.chroma_errors += 1
                else:
                    self.metrics.ollama_errors += 1
                log.exception("Task %s failed after %.1fms", task_id, latency_ms)
                return self._task_error(task_id, "task_failed", str(exc), status=502)
            finally:
                self.metrics.active_tasks = max(0, self.metrics.active_tasks - 1)

    def _task_error(
        self,
        task_id: str,
        code: str,
        message: str,
        *,
        status: int,
    ) -> web.Response:
        return web.json_response(
            {
                "task_id": task_id,
                "status": "failed",
                "worker_id": self.config.worker_id,
                "error": {"code": code, "message": message},
            },
            status=status,
        )

    async def register_with_master(self) -> None:
        if not self.config.master_url:
            log.info("MASTER_URL not set; skipping worker registration")
            return
        payload = {
            "worker_id": self.config.worker_id,
            "host": self.config.advertise_host,
            "port": self.config.port,
        }
        # CHANGED: retry up to 10 times with 3 second delay
        for attempt in range(10):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.master_url}/register",
                        json=payload,
                        timeout=5
                    ) as resp:
                        if resp.status < 300:
                            log.info("Registered worker %s with master", self.config.worker_id)
                            return
                        else:
                            log.warning("Master registration returned status %d", resp.status)
            except Exception as exc:
                log.warning(
                    "Master registration attempt %d/10 failed: %s — retrying in 3s",
                    attempt + 1, exc
                )
                await asyncio.sleep(3)
        log.warning("Could not register with master after 10 attempts — continuing anyway")

    async def heartbeat_loop(self) -> None:
        if not self.config.master_url:
            return
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.sleep(self.config.heartbeat_interval_s)
                payload = {
                    "worker_id": self.config.worker_id,
                    "load": self.load,
                    "active_tasks": self.metrics.active_tasks,
                    "max_concurrency": self.config.max_concurrency,
                    "total_tasks": self.metrics.total_tasks,
                }
                try:
                    async with session.post(
                        f"{self.config.master_url}/heartbeat",
                        json=payload,
                        timeout=5,
                    ) as resp:
                        if resp.status >= 300:
                            log.warning("Heartbeat returned status %d", resp.status)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("Heartbeat failed: %s", exc)

    async def on_startup(self, app: web.Application) -> None:
        await self.register_with_master()
        self._heartbeat_task = asyncio.create_task(self.heartbeat_loop())

    async def on_cleanup(self, app: web.Application) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)

    def make_app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.get("/metrics", self.handle_metrics),
                web.post("/task", self.handle_task),
            ]
        )
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app


def source_to_response(source: SourceSnippet) -> Dict[str, Any]:
    response = {
        "source_file": source.source_file,
        "page": source.page,
        "chunk_id": source.chunk_id,
    }
    if source.score is not None:
        response["score"] = round(source.score, 4)
    return response


def main() -> None:
    config = WorkerConfig.from_env()
    worker = GPUWorker(config)
    log.info("Starting worker %s on %s:%d", config.worker_id, config.host, config.port)
    web.run_app(worker.make_app(), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
