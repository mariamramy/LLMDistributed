import asyncio
import aiohttp
import logging
from typing import List, Optional
from lb.node import Node  

log = logging.getLogger("load_balancer")

# Health Monitor Class
class HealthMonitor:
    #parameters: list of nodes managed, interval_s how often to run health checks, and HTTP timeout
    def __init__(self, nodes: List[Node], interval_s: float = 5.0, timeout_s: float = 2.0):
        self.nodes = nodes
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._task: Optional[asyncio.Task] = None

    async def _check_node(self, session: aiohttp.ClientSession, node: Node):
        try:
            async with session.egt(
                node.health_url,
                # if the node doesn't respond within 2 seconds, it raises an exception and falls into the except block 
                timeout= aiohttp.ClientTimeout(total=self.timeout_s),
            ) as resp:
                #if node replies with json file
                if resp.status == 200:
                    data = await resp.json()
                    node.healthy = True
                    node.load = float(data.get("load", 0.0))
                    log.debug("Node %s healthy (load=%.2f)", node.node_id, node.load)
                
                else:
                    node.healthy = False
                    log.warning("Node %s returned status %d", node.node_id, resp.status)
        
        #if the node didnt respond after timeout
        except Exception as exc:
            if node.healthy:
                log.warning("Node %s unreachable: %s — marking unhealthy", node.node_id, exc)

            node.healthy = False

    async def _loop(self):
        connector = aiohttp.TCPConnector(limit=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                await asyncio.gather(
                    *[self._check_node(session, n) for n in self.nodes]
                )
                await asyncio.sleep(self.interval_s)

    def start(self):
        self._task = asyncio.ensure_future(self._loop())

    def stop(self):
        if self._task:
            self._task.cancel()