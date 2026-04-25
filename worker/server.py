from __future__ import annotations

import asyncio
import os
import time

import grpc
import worker_pb2
import worker_pb2_grpc
from grpc import aio

from worker.llm import run_llm
from worker.rag import retrieve_context


WORKER_ID = os.environ.get("WORKER_ID", "worker-unknown")
PORT = int(os.environ.get("GRPC_PORT", "50051"))

active_requests = 0
recent_latencies: list[float] = []


class WorkerServicer(worker_pb2_grpc.WorkerServiceServicer):
    async def Process(self, request, context):
        global active_requests, recent_latencies

        active_requests += 1
        start = time.perf_counter()
        try:
            print(
                f"[{WORKER_ID}] Processing request {request.id}: "
                f"{request.query[:80]}",
                flush=True,
            )

            rag_context = await retrieve_context(request.query)
            result = await run_llm(request.query, rag_context)

            latency = (time.perf_counter() - start) * 1000
            recent_latencies.append(latency)
            if len(recent_latencies) > 100:
                recent_latencies = recent_latencies[-100:]

            return worker_pb2.QueryResponse(
                id=request.id,
                result=result,
                latency=latency,
                worker_id=WORKER_ID,
            )
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return worker_pb2.QueryResponse(
                id=request.id,
                result="",
                latency=(time.perf_counter() - start) * 1000,
                worker_id=WORKER_ID,
            )
        finally:
            active_requests = max(0, active_requests - 1)

    async def HealthCheck(self, request, context):
        avg_latency = (
            sum(recent_latencies) / len(recent_latencies)
            if recent_latencies
            else 0.0
        )
        return worker_pb2.HealthResponse(
            healthy=True,
            active_requests=active_requests,
            avg_latency_ms=avg_latency,
        )


async def serve() -> None:
    server = aio.server()
    worker_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    await server.start()
    print(f"[{WORKER_ID}] gRPC server listening on port {PORT}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
