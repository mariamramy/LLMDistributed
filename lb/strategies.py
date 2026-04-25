import threading
from typing import Protocol, Sequence


class WorkerLike(Protocol):
    healthy: bool
    active_connections: int
    avg_latency_ms: float


class RoundRobin:
    def __init__(self, workers: Sequence[WorkerLike]):
        self.workers = workers
        self.index = 0
        self._lock = threading.Lock()

    def pick(self) -> WorkerLike:
        if not self.workers:
            raise RuntimeError("No workers registered")
        with self._lock:
            worker = self.workers[self.index % len(self.workers)]
            self.index += 1
            return worker


class LeastConnections:
    def __init__(self, workers: Sequence[WorkerLike]):
        self.workers = workers

    def pick(self) -> WorkerLike:
        if not self.workers:
            raise RuntimeError("No workers registered")
        return min(
            self.workers,
            key=lambda worker: (
                worker.active_connections if worker.healthy else float("inf")
            ),
        )


class LoadAware:
    """Composite score: active connections plus normalized recent latency."""

    def __init__(self, workers: Sequence[WorkerLike]):
        self.workers = workers

    def pick(self) -> WorkerLike:
        if not self.workers:
            raise RuntimeError("No workers registered")

        def score(worker: WorkerLike) -> float:
            if not worker.healthy:
                return float("inf")
            return worker.active_connections + (worker.avg_latency_ms / 1000)

        return min(self.workers, key=score)
