from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

import grpc
import worker_pb2
from fastapi import FastAPI, HTTPException
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from lb.strategies import LeastConnections, LoadAware, RoundRobin
from lb.worker_registry import WorkerProxy, parse_worker_hosts


ACTIVE_CONNS = Gauge(
    "worker_active_connections",
    "Active gRPC calls currently assigned to each worker",
    ["worker_id"],
)
WORKER_HEALTH = Gauge(
    "worker_health",
    "Worker health status reported by gRPC health checks",
    ["worker_id"],
)
WORKER_AVG_LATENCY = Gauge(
    "worker_avg_latency_ms",
    "Worker-reported rolling average request latency in milliseconds",
    ["worker_id"],
)

STRATEGIES = {
    "round_robin": RoundRobin,
    "least_conn": LeastConnections,
    "load_aware": LoadAware,
}

workers: list[WorkerProxy] = []
strategy: RoundRobin | LeastConnections | LoadAware | None = None
health_task: asyncio.Task[None] | None = None


class QueryPayload(BaseModel):
    id: int = 0
    query: str = Field(..., min_length=1)


def _set_worker_metrics(worker: WorkerProxy) -> None:
    ACTIVE_CONNS.labels(worker_id=worker.id).set(worker.active_connections)
    WORKER_HEALTH.labels(worker_id=worker.id).set(1 if worker.healthy else 0)
    WORKER_AVG_LATENCY.labels(worker_id=worker.id).set(worker.avg_latency_ms)


async def health_watchdog(interval_seconds: int = 5) -> None:
    while True:
        for worker in workers:
            try:
                response = await worker.stub.HealthCheck(
                    worker_pb2.HealthRequest(),
                    timeout=2,
                )
                worker.healthy = response.healthy
                worker.active_connections = response.active_requests
                worker.avg_latency_ms = response.avg_latency_ms
            except grpc.RpcError:
                worker.healthy = False
            finally:
                _set_worker_metrics(worker)
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workers, strategy, health_task

    raw_hosts = os.environ.get(
        "WORKER_HOSTS",
        "localhost:50051,localhost:50052,localhost:50053",
    )
    workers = parse_worker_hosts(raw_hosts)

    mode = os.environ.get("LB_STRATEGY", "round_robin")
    strategy_class = STRATEGIES.get(mode)
    if strategy_class is None:
        valid_modes = ", ".join(sorted(STRATEGIES))
        raise RuntimeError(f"Unknown LB_STRATEGY '{mode}'. Use one of: {valid_modes}.")
    strategy = strategy_class(workers)

    for worker in workers:
        _set_worker_metrics(worker)

    health_task = asyncio.create_task(health_watchdog())
    try:
        yield
    finally:
        if health_task:
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
        await asyncio.gather(*(worker.close() for worker in workers))


app = FastAPI(title="Distributed LLM Load Balancer", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health() -> dict:
    return {
        "healthy": any(worker.healthy for worker in workers),
        "workers": [
            {
                "id": worker.id,
                "healthy": worker.healthy,
                "active_connections": worker.active_connections,
                "avg_latency_ms": worker.avg_latency_ms,
            }
            for worker in workers
        ],
    }


@app.post("/query")
async def query(payload: QueryPayload) -> dict:
    if strategy is None or not workers:
        raise HTTPException(status_code=503, detail="No workers registered")

    attempts = 0
    max_attempts = len(workers)
    last_error = "All workers unavailable"

    while attempts < max_attempts:
        attempts += 1
        worker = strategy.pick()
        if not worker.healthy:
            continue

        request_was_assigned = False
        try:
            worker.active_connections += 1
            request_was_assigned = True
            _set_worker_metrics(worker)

            start = time.perf_counter()
            grpc_request = worker_pb2.QueryRequest(id=payload.id, query=payload.query)
            response = await worker.stub.Process(grpc_request, timeout=30)

            worker.avg_latency_ms = (time.perf_counter() - start) * 1000
            _set_worker_metrics(worker)

            return {
                "id": response.id,
                "result": response.result,
                "latency": response.latency,
                "served_by": response.worker_id,
            }
        except grpc.RpcError as exc:
            worker.healthy = False
            last_error = exc.details() or exc.code().name
            _set_worker_metrics(worker)
        finally:
            if request_was_assigned:
                worker.active_connections = max(0, worker.active_connections - 1)
                _set_worker_metrics(worker)

    raise HTTPException(status_code=503, detail=last_error)
