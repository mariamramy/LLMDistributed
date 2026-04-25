from dataclasses import dataclass


@dataclass
class WorkerSnapshot:
    worker_id: str
    healthy: bool
    active_connections: int
    avg_latency_ms: float
