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
         