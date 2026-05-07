import asyncio
import aiohttp
from aiohttp import web
import argparse
import logging
import time
from typing import List

from lb.node import Node                          
from lb.strategies import BaseStrategy, RoundRobinStrategy, STRATEGIES  
from lb.health_monitor import HealthMonitor  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LB] %(levelname)s %(message)s",
)
log = logging.getLogger("load_balancer")

#Load Balancer Class
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


# used when as a fallback when forward fails mid-request. It directly forwards to the specified node without using the strategy, and it does not do retries if it fails.
# sends directly to a node 
async def _forward_to(self, session, node: Node, payload: dict) -> web.Response:
    node.active_connections += 1
    try:
        async with session.post(
            node.request_url, json=payload,
            timeout=aiohttp.ClientTimeout(total=self.forward_timeout_s), ##sends the apyload to the master scheduler with a timeout, if the request takes longer than the timeout, it will raise an exception
        ) as resp: # captures the response from opening the connection from the master scheduler, if the master scheduler returns a non-200 status code, it will raise an exception
            data = await resp.json()
            return web.json_response(data, status=resp.status) # status from node,200 OK, 500 Internal Server Error, etc
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)
    finally:
        node.active_connections = max(0, node.active_connections - 1)  # undo the increment from the top, the request is done


# ROUTE HANDLERS !!


async def handle_request(self, request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    async with aiohttp.ClientSession() as session:
        return await self._forward(session, payload)

async def handle_health(self, _: web.Request) -> web.Response:
    nodes_info = [
        {"node_id": n.node_id, "url": n.url, "healthy": n.healthy,
         "active_connections": n.active_connections, "load": round(n.load, 3),
         "total_requests": n.total_requests}
        for n in self.nodes
    ]
    return web.json_response({"status": "ok", "total_requests": self._total_requests,
                               "total_failures": self._total_failures,
                               "strategy": type(self.strategy).__name__, "nodes": nodes_info})

async def handle_stats(self, _: web.Request) -> web.Response:
    healthy_count = sum(1 for n in self.nodes if n.healthy)
    return web.json_response({"healthy_nodes": healthy_count, "total_nodes": len(self.nodes),
                               "total_requests": self._total_requests, "total_failures": self._total_failures,
                               "strategy": type(self.strategy).__name__})




# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Cluster — Load Balancer")
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--strategy", default="round_robin",
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--master-host", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=9000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nodes = [Node(node_id="master-1", host=args.master_host, port=args.master_port)]
    lb = LoadBalancer(
        nodes=nodes,
        strategy=STRATEGIES[args.strategy],
        host=args.host,
        port=args.port,
    )
    lb.run()