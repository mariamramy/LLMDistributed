import asyncio
import aiohttp
from aiohttp import web
import argparse
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LB] %(levelname)s %(message)s",
)
log = logging.getLogger("load_balancer")


# defining a node 
@dataclass
class Node:
    node_id: str
    host: str 
    port: int # port the node listens on
    healthy: bool = True # alive or not
    active_connections: int = 0 # how many requests its currently handling
    load: float = 0.0 # a score from 0.0-1.0 reported from node used in load aware
    total_requests: int = 0 # for monitoring

    #properties auto-build the HTTP addresses
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.url}/health"

    @property
    def request_url(self) -> str:
        return f"{self.url}/request"



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