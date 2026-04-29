import threading
from abc import ABC, abstractmethod
from typing import List, Optional
from lb.node import Node  

# a template that other strategies inherits from
class BaseStrategy(ABC):
    @abstractmethod
    def select(self, nodes: List[Node]) -> Optional[Node]: ...

    def _healthy(self, nodes: List[Node]) -> List[Node]:
        return [n for n in nodes if n.healthy]

# alternates between nodes equally, doesn't care if a node has many requests and the other is idle
class RoundRobinStrategy(BaseStrategy):
    def __init__(self):
        self._index = 0 # tracks whos turn it is
        self._lock = threading.Lock()

    def select(self, nodes: List[Node]) -> Optional[Node]:
        healthy = self._healthy(nodes)
        if not healthy:
            return None
        with self._lock:
            node = healthy[self._index % len(healthy)]
            self._index += 1
        return node


# scans all healthy nodes and returns the node with the smallest active connections
class LeastConnectionsStrategy(BaseStrategy):
    def select(self, nodes: List[Node]) -> Optional[Node]:
        healthy = self._healthy(nodes)
        if not healthy:
            return None
        
        return min(healthy, key=lambda n: n.active_connections)

# same as least connections but incorperates node load
class LoadAwareStrategy(BaseStrategy):
    def select(self, nodes: List[Node]) -> Optional[Node]:
        healthy = self._healthy(nodes)
        if not healthy:
            return None
        
        # node.load comes from the health check report, 0.0-1.0 score, choose the node with the least
        return min(healthy, key=lambda n: (n.load, n.active_connections))


STRATEGIES = {
    "round_robin":       RoundRobinStrategy(),
    "least_connections": LeastConnectionsStrategy(),
    "load_aware":        LoadAwareStrategy(),
}