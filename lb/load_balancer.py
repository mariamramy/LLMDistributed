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