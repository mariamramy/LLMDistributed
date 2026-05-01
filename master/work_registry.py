import logging
import time
from typing import Dict, List, Optional
from master.models import WorkerInfo, TaskStatus, Task, WORKER_TIMEOUT_S

log = logging.getLogger("master_scheduler")

# a database of all active GPU compute nodes in the cluster
class WorkerRegistry:
 
    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}

    # node entry 
    def register(self, worker_id: str, host: str, port: int) -> WorkerInfo:
        worker = WorkerInfo(worker_id=worker_id, host=host, port=port)
        self._workers[worker_id] = worker
        log.info("Worker registered: %s @ %s:%d", worker_id, host, port)
        return worker
    
    # node exiting
    def deregister(self, worker_id: str):
        if worker_id in self._workers:
            del self._workers[worker_id]
            log.info("Worker deregistered: %s", worker_id)

    # Updates the last_heartbeat timestamp and the current load of a worker
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

    # find the worker with the lowest load and fewest active tasks
    def get_best_worker(self) -> Optional[WorkerInfo]:

        # return the healthy worker with the lowest combined load score
        healthy = self.get_healthy_workers()
        if not healthy:
            return None
        return min(healthy, key=lambda w: (w.load, w.active_tasks))

    def get_all(self) -> List[WorkerInfo]:
        return list(self._workers.values())

    # It identifies workers that haven't sent a heartbeat within WORKER_TIMEOUT_S
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
    
class TaskStore:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def add(self, task: Task):
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def update(self, task: Task):
        self._tasks[task.task_id] = task

    # If a worker fails, this finds which tasks were running on that specific node so they can be reassigned to active nodes
    def get_in_flight_by_worker(self, worker_id: str) -> List[Task]:
        return [
            t for t in self._tasks.values()
            if t.status == TaskStatus.IN_FLIGHT and t.assigned_worker == worker_id
        ]

    # summary of task statuses 
    def summary(self) -> dict:
        counts = {s: 0 for s in TaskStatus}
        for t in self._tasks.values():
            counts[t.status] += 1
        return {s.value: n for s, n in counts.items()}