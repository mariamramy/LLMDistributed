from __future__ import annotations

from dataclasses import dataclass, field

import grpc

import worker_pb2_grpc


@dataclass
class WorkerProxy:
    host: str
    port: int
    healthy: bool = True
    active_connections: int = 0
    avg_latency_ms: float = 0.0
    id: str = field(init=False)
    channel: grpc.aio.Channel = field(init=False)
    stub: worker_pb2_grpc.WorkerServiceStub = field(init=False)

    def __post_init__(self) -> None:
        self.id = f"{self.host}:{self.port}"
        self.channel = grpc.aio.insecure_channel(self.id)
        self.stub = worker_pb2_grpc.WorkerServiceStub(self.channel)

    async def close(self) -> None:
        await self.channel.close()


def parse_worker_hosts(raw_hosts: str) -> list[WorkerProxy]:
    workers: list[WorkerProxy] = []
    for raw_host in raw_hosts.split(","):
        address = raw_host.strip()
        if not address:
            continue
        host, separator, port = address.partition(":")
        if not separator or not host or not port:
            raise ValueError(
                f"Invalid worker address '{address}'. Expected format host:port."
            )
        workers.append(WorkerProxy(host=host, port=int(port)))
    return workers
