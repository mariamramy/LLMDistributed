import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(str, Enum):
    PENDING    = "pending"
    IN_FLIGHT  = "in_flight"
    COMPLETED  = "completed"
    FAILED     = "failed"


WORKER_TIMEOUT_S   = 15.0   # seconds before a worker is considered dead
DISPATCH_INTERVAL  = 0.05   # seconds between dispatch-loop iterations
HEARTBEAT_INTERVAL = 5.0    # expected heartbeat frequency from workers
MAX_TASK_RETRIES   = 3      # max requeue attempts before marking task FAILED


@dataclass
class Task:
    # Represents one user request flowing through the scheduling pipeline.
    
    task_id: str             = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str          = ""
    payload: dict            = field(default_factory=dict)
    status: TaskStatus       = TaskStatus.PENDING
    assigned_worker: str     = ""
    priority: int            = 5
    created_at: float        = field(default_factory=time.time)
    started_at: float        = 0.0
    completed_at: float      = 0.0
    retries: int             = 0
    result: dict             = field(default_factory=dict)

    # Make Task orderable for PriorityQueue: (priority, created_at, Task)
    def __lt__(self, other: "Task") -> bool:
        return (self.priority, self.created_at) < (other.priority, other.created_at)


@dataclass
class WorkerInfo:

    worker_id: str
    host: str
    port: int
    healthy: bool       = True
    load: float         = 0.0
    active_tasks: int   = 0
    total_tasks: int    = 0
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def task_url(self) -> str:
        return f"{self.url}/task"