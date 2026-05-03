import asyncio
from urllib import request
import aiohttp
from aiohttp import web
import argparse
import logging
import time
from typing import List

from lb import node
from lb.node import Node                          
from lb.strategies import BaseStrategy, RoundRobinStrategy, STRATEGIES  
from lb.health_monitor import HealthMonitor  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LB] %(levelname)s %(message)s",
)
log = logging.getLogger("load_balancer")

# Load Balancer Class
class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0

    def get_next_worker(self):
        worker = self.workers[self.index]
        self.index = (self.index + 1) % len(self.workers)
        return worker

    def dispatch(self, request):
        worker = self.get_next_worker()
        return worker.process(request)
    
    def __init__ (
            self, nodes: List[Node],
            strategy: BaseStrategy = None, 
            host: str = '0.0.0.0',
            port: int = 8080,
            health_interval_s: float = 5.0,
            forward_timeout_s: int = 30
    ):
        self.nodes = nodes
        self.strategy = strategy or RoundRobinStrategy()
        self.host = host
        self.port = port
        self.health_monitor = HealthMonitor(nodes, health_interval_s)
        self.forward_timeout_s = forward_timeout_s
        self._total_requests = 0
        self._total_failure = 0

    async def _forward(self, session: aiohttp.ClientSession, payload: dict) -> web.Response:
         node = self.strategy.select(self.nodes)

         if node is None:
             self._total_failure += 1
             return web.json_response({"error": "Service unavailable — no healthy nodes"}, status=503)
         
         node.active_connections += 1
         node.total_requests += 1
         self._total_requests += 1
         start = time.perf_counter()

#forwards the payload to the selected node. It first checks if a node is available using the strategy, and if not, it returns a 503 error. If a node is selected, it increments the active connection count and total request count for that node and the load balancer. It then forwards the request to the selected node using an aiohttp client session, measuring the latency and logging it. If the request fails, it marks the node as unhealthy and tries one retry on a different node before returning an error response. Finally, it decrements the active connection count for the node.
#  asks the strategy to pick a node then increments the active connection count and total request count for that node and the load balancer. It then forwards the request to the selected node using an aiohttp client session, measuring the latency and logging it. 
# If the request fails, it marks the node as unhealthy and tries one retry on a different node before returning an error response. Finally, it decrements the active connection count for the node.
async def _forward(self, session: aiohttp.ClientSession, payload: dict) -> web.Response:
    node = self.strategy.select(self.nodes)

    if node is None:
        self._total_failures += 1
        return web.json_response({"error": "Service unavailable — no healthy nodes"}, status=503)

    node.active_connections += 1
    node.total_requests += 1
    self._total_requests += 1
    start = time.perf_counter()

    try:
        async with session.post(
            node.request_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.forward_timeout_s),
        ) as resp:
            data = await resp.json()
            elapsed = (time.perf_counter() - start) * 1000
            log.info("→ node=%s latency=%.1fms status=%d", node.node_id, elapsed, resp.status)
            return web.json_response(data, status=resp.status)

    except Exception as exc:
        node.healthy = False  # mark it bad so health monitor re-checks it
        self._total_failures += 1

        # one retry on a different node
        retry_node = self.strategy.select(self.nodes)
        if retry_node and retry_node.node_id != node.node_id:
            return await self._forward_to(session, retry_node, payload)

        return web.json_response({"error": f"Node {node.node_id} failed: {exc}"}, status=502)

    finally:
        node.active_connections = max(0, node.active_connections - 1)

