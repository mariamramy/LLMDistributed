from dataclasses import dataclass, field

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
