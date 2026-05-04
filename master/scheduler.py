import asyncio
import aiohttp
from aiohttp import web
import argparse
import logging
import time
import uuid
from typing import Dict

from master.models import Task, TaskStatus, WorkerInfo, DISPATCH_INTERVAL, HEARTBEAT_INTERVAL, MAX_TASK_RETRIES
from master.work_registry import WorkerRegistry, TaskStore

log = logging.getLogger("master_scheduler")

class MasterScheduler:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        dispatch_interval: float = DISPATCH_INTERVAL,
        heartbeat_check_interval: float = HEARTBEAT_INTERVAL,
        forward_timeout_s: int = 30,
    ):
        self.host = host
        self.port = port
        self.dispatch_interval = dispatch_interval
        self.heartbeat_check_interval = heartbeat_check_interval
        self.forward_timeout_s = forward_timeout_s

        # Every incoming request gets wrapped in a Task and dropped onto this queue
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._registry = WorkerRegistry() # The scheduler talks to this whenever it needs to know which workers are alive
        self._store = TaskStore()

        # create a Future to store while HTTP handler waits for the result 
        self._pending: Dict[str, asyncio.Future] = {}

        # Stats
        self._dispatched = 0
        self._completed  = 0
        self._failed     = 0
    
    # put task in priority queue
    async def _enqueue(self, task: Task):
        # save task in TaskStore
        self._store.add(task)
        
        # queue looks at priorty first then sees when task was created (incase they have the same priorty)
        await self._queue.put((task.priority, task.created_at, task))
        log.debug("Enqueued task=%s priority=%d", task.task_id[:8], task.priority)

    # sends a task to a GPU worker over HTTP
    async def _dispatch_to_worker(
        self, session: aiohttp.ClientSession, worker: WorkerInfo, task: Task) -> bool:
        
        # update state before sending
        task.status = TaskStatus.IN_FLIGHT
        task.assigned_worker = worker.worker_id # assigned to which worker
        task.started_at = time.time() 
        self._store.update(task)
        self._registry.update_task_count(worker.worker_id, +1)
        worker.total_tasks += 1

        try:

            # send the HTTP request
            async with session.post(
                worker.task_url,
                json={"task_id": task.task_id, **task.payload},
                timeout=aiohttp.ClientTimeout(total=self.forward_timeout_s),
            ) as resp:
                if resp.status == 200:
                    log.info(
                        "Dispatched task=%s → worker=%s",
                        task.task_id[:8], worker.worker_id,
                    )
                    self._dispatched += 1
                    return True
                else:
                    log.warning(
                        "Worker %s rejected task=%s status=%d",
                        worker.worker_id, task.task_id[:8], resp.status,
                    )
                    return False
                
        except Exception as exc:
            log.warning(
                "Dispatch to worker=%s failed for task=%s: %s",
                worker.worker_id, task.task_id[:8], exc,
            )
            worker.healthy = False
            return False
        # always decrements the worker's active task count by -1 
        finally:
            self._registry.update_task_count(worker.worker_id, -1)


    # FAULT TOLERANCE
    # Move all in-flight tasks of a dead worker back to the queue
    async def _reassign_worker_tasks(self, worker_id: str):

        # finds every task that was IN_FLIGHT and assigned to the dead worker
        orphans = self._store.get_in_flight_by_worker(worker_id)
        for task in orphans:
            
            # If this task has already been retried 3 times and keeps failing then its marked as FAILED
            if task.retries >= MAX_TASK_RETRIES:
                task.status = TaskStatus.FAILED
                self._store.update(task)
                self._failed += 1
                log.error(
                    "Task=%s exceeded max retries — marked FAILED", task.task_id[:8]
                )

                # Resolve the future with an error so the LB gets a response and doesnt keep waiting
                fut = self._pending.pop(task.task_id, None)
                if fut and not fut.done():
                    fut.set_result({"error": "Task failed after max retries", "task_id": task.task_id})
            
            else:
                task.retries += 1
                task.status = TaskStatus.PENDING # back to waiting
                task.assigned_worker = "" # unassigned, so the dispatch loop can freely give it to any healthy worker
                self._store.update(task)
                log.info(
                    "Re-queuing task=%s (retry %d/%d)",
                    task.task_id[:8], task.retries, MAX_TASK_RETRIES,
                )
                await self._queue.put((task.priority, task.created_at, task))


    # BACKGROUND LOOP
    # dequeues tasks and forwards them to available workers
    async def _dispatch_loop(self):
        
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                try:
                    # skip if no tasks
                    if self._queue.empty():
                        await asyncio.sleep(self.dispatch_interval)
                        continue
                    
                    # are there any alive workers?
                    worker = self._registry.get_best_worker()
                    if worker is None:
                        log.warning("No healthy workers — waiting...")
                        await asyncio.sleep(1.0)
                        continue

                    # Block until a task is available
                    _, _, task = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )

                    success = await self._dispatch_to_worker(session, worker, task)
                    
                    # if function dispatch to worker returned false
                    if not success:
                        # Re-queue if dispatch failed
                        if task.retries < MAX_TASK_RETRIES:
                            task.retries += 1
                            task.status = TaskStatus.PENDING
                            await self._queue.put((task.priority, task.created_at, task))
                        else:
                            task.status = TaskStatus.FAILED
                            self._store.update(task)
                            self._failed += 1
                            fut = self._pending.pop(task.task_id, None)
                            if fut and not fut.done():
                                fut.set_result({"error": "Dispatch failed"})

                except asyncio.TimeoutError:
                    pass  # Queue was empty, loop again
                except Exception as exc:
                    log.error("Dispatch loop error: %s", exc)
                    await asyncio.sleep(self.dispatch_interval)

    # Periodically check for timed-out workers and reassign their tasks
    async def _heartbeat_check_loop(self):
        
        while True:
            dead_ids = self._registry.check_timeouts()
            for wid in dead_ids:
                await self._reassign_worker_tasks(wid)
            await asyncio.sleep(self.heartbeat_check_interval)



    # -------------------------------------------------------------------------
    # HTTP Route Handlers
    # -------------------------------------------------------------------------


    # recieves from the lb, wraps the payload in a Task, puts it on the queue, and waits for the result Future to be resolved by the worker result handler. If the Future is not resolved within forward_timeout_s, it returns a 504 error to the client and marks the task as FAILED.
    async def handle_request(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        task = Task(
            request_id=payload.get("request_id", str(uuid.uuid4())),
            payload=payload,
            priority=payload.get("priority", 5),
        )

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[task.task_id] = fut

        await self._enqueue(task)

        log.info("Accepted request=%s → task=%s (queue_size=%d)",
                 task.request_id[:8], task.task_id[:8], self._queue.qsize())

        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=self.forward_timeout_s)
            return web.json_response(result)
        except asyncio.TimeoutError:
            self._pending.pop(task.task_id, None)
            task.status = TaskStatus.FAILED
            self._store.update(task)
            return web.json_response({"error": "Request timed out", "task_id": task.task_id}, status=504)



    # reports on the master's state; how many workers are alive, how big the queue is, what the load is
    async def handle_worker_result(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        task_id = data.get("task_id")
        task = self._store.get(task_id)

        if not task:
            return web.json_response({"error": "Unknown task_id"}, status=404)

        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = data
        self._store.update(task)
        self._completed += 1

        log.info("Result received for task=%s from worker=%s (%.1fms)",
                 task_id[:8], data.get("worker_id", "?"),
                 (task.completed_at - task.started_at) * 1000)

        fut = self._pending.pop(task_id, None)
        if fut and not fut.done():
            fut.set_result(data)

        return web.json_response({"status": "ok"})

    async def handle_register(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        worker_id = data.get("worker_id") or str(uuid.uuid4())
        host      = data.get("host", request.remote)
        port      = int(data.get("port", 9100))

        self._registry.register(worker_id, host, port)
        return web.json_response({"status": "registered", "worker_id": worker_id})

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        worker_id = data.get("worker_id")
        load      = float(data.get("load", 0.0))

        if not worker_id:
            return web.json_response({"error": "Missing worker_id"}, status=400)

        self._registry.heartbeat(worker_id, load)
        return web.json_response({"status": "ok"})

    async def handle_health(self, _: web.Request) -> web.Response:
        healthy_workers = len(self._registry.get_healthy_workers())
        total_workers   = len(self._registry.get_all())
        queue_size      = self._queue.qsize()

        max_capacity = max(total_workers * 10, 1)
        load = min(1.0, queue_size / max_capacity)

        return web.json_response({
            "status":          "ok",
            "load":            round(load, 3),
            "healthy_workers": healthy_workers,
            "total_workers":   total_workers,
            "queue_size":      queue_size,
        })

    async def handle_stats(self, _: web.Request) -> web.Response:
        workers = [
            {
                "worker_id":      w.worker_id,
                "url":            w.url,
                "healthy":        w.healthy,
                "load":           round(w.load, 3),
                "active_tasks":   w.active_tasks,
                "total_tasks":    w.total_tasks,
                "last_heartbeat": round(time.time() - w.last_heartbeat, 1),
            }
            for w in self._registry.get_all()
        ]
        return web.json_response({
            "tasks":      self._store.summary(),
            "dispatched": self._dispatched,
            "completed":  self._completed,
            "failed":     self._failed,
            "queue_size": self._queue.qsize(),
            "workers":    workers,
        })

    async def handle_deregister(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        worker_id = data.get("worker_id")
        if worker_id:
            await self._reassign_worker_tasks(worker_id)
            self._registry.deregister(worker_id)

        return web.json_response({"status": "deregistered"})

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def run(self):
        app = web.Application()
        app.router.add_post("/request",    self.handle_request)
        app.router.add_post("/result",     self.handle_worker_result)
        app.router.add_post("/register",   self.handle_register)
        app.router.add_post("/heartbeat",  self.handle_heartbeat)
        app.router.add_post("/deregister", self.handle_deregister)
        app.router.add_get("/health",      self.handle_health)
        app.router.add_get("/stats",       self.handle_stats)

        async def on_startup(_app):
            asyncio.ensure_future(self._dispatch_loop())
            asyncio.ensure_future(self._heartbeat_check_loop())
            log.info("Master Scheduler started on %s:%d", self.host, self.port)

        app.on_startup.append(on_startup)
        web.run_app(app, host=self.host, port=self.port, print=None)