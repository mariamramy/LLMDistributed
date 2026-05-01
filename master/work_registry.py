import logging
import time
from typing import Dict, List, Optional
from master.models import WorkerInfo, WORKER_TIMEOUT_S

log = logging.getLogger("master_scheduler")


class WorkerRegistry:
 
    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}

    # CRUD 
    def register(self, worker_id: str, host: str, port: int) -> WorkerInfo:
        worker = WorkerInfo(worker_id=worker_id, host=host, port=port)
        self._workers[worker_id] = worker
        log.info("Worker registered: %s @ %s:%d", worker_id, host, port)
        return worker

    def deregister(self, worker_id: str):
        if worker_id in self._workers:
            del self._workers[worker_id]
            log.info("Worker deregistered: %s", worker_id)

    def heartbeat(self, worker_id: str, load: float = 0.0):
        if worker_id in self._workers:
            w = self._workers[worker_id]
            w.last_heartbeat = time.time()
            w.healthy = True
            w.load = load

    def update_task_count(self, worker_id: str, delta: int):
        if worker_id in self._workers:
            self._workers[worker_id].active_tasks = max(
                0, self._workers[worker_id].active_tasks + delta
            )


    # Queries
    def get_healthy_workers(self) -> List[WorkerInfo]:
        return [w for w in self._workers.values() if w.healthy]

    def get_best_worker(self) -> Optional[WorkerInfo]:

        # return the healthy worker with the lowest combined load score
        healthy = self.get_healthy_workers()
        if not healthy:
            return None
        return min(healthy, key=lambda w: (w.load, w.active_tasks))

    def get_all(self) -> List[WorkerInfo]:
        return list(self._workers.values())

    def check_timeouts(self) -> List[str]:

        # mark workers as unhealthy if they missed their heartbeat window
        now = time.time()
        dead = []
        for w in self._workers.values():
            if w.healthy and (now - w.last_heartbeat) > WORKER_TIMEOUT_S:
                w.healthy = False
                dead.append(w.worker_id)
                log.warning("Worker %s timed out — marked unhealthy", w.worker_id)
        return dead