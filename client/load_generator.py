import asyncio
import aiohttp
import argparse
import time
import random
import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT] %(levelname)s %(message)s",
)
log = logging.getLogger("client")

# ---------------------------------------------------------------------------
# Sample prompts to mimic real workloads
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Summarise the key events of World War II.",
    "Write a short Python function to reverse a linked list.",
    "What are the main differences between TCP and UDP?",
    "Describe how transformers work in natural language processing.",
    "List five best practices for REST API design.",
    "Explain gradient descent and its variants.",
    "What is retrieval-augmented generation (RAG)?",
    "How does CUDA enable GPU parallel computing?",
    "Describe the CAP theorem in distributed systems.",
    "Compare round-robin and least-connections load balancing.",
    "What are the benefits of using vector databases?",
    "Explain fault tolerance strategies in distributed systems.",
    "How does NGINX handle load balancing?",
    "What is the difference between horizontal and vertical scaling?",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Request:
    """Represents a single user request sent to the load balancer."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: int = 0
    prompt: str = ""
    use_rag: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequestResult:
    """Stores the outcome of a single request."""
    request_id: str
    user_id: int
    success: bool
    latency_ms: float
    status_code: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Core client logic
# ---------------------------------------------------------------------------
class LoadGenerator:

    def __init__(
        self,
        lb_url: str,
        num_users: int = 1000,
        requests_per_user: int = 1,
        think_time_ms: int = 0,
        timeout_s: int = 30,
    ):
        self.lb_url = lb_url
        self.num_users = num_users
        self.requests_per_user = requests_per_user
        self.think_time_ms = think_time_ms
        self.timeout_s = timeout_s
        self.results: List[RequestResult] = []

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def _build_request(self, user_id: int) -> Request:
        """Build a randomised request payload for a given virtual user."""
        return Request(
            user_id=user_id,
            prompt=random.choice(SAMPLE_PROMPTS),
            use_rag=random.choice([True, False]),
        )

    async def _send_request(
        self, session: aiohttp.ClientSession, req: Request
    ) -> RequestResult:
        """Send one HTTP POST request and record the outcome."""
        start = time.perf_counter()
        try:
            async with session.post(
                self.lb_url,
                json=req.to_dict(),
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            ) as resp:
                await resp.json()          # consume body
                latency_ms = (time.perf_counter() - start) * 1000
                return RequestResult(
                    request_id=req.request_id,
                    user_id=req.user_id,
                    success=resp.status == 200,
                    latency_ms=latency_ms,
                    status_code=resp.status,
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                request_id=req.request_id,
                user_id=req.user_id,
                success=False,
                latency_ms=latency_ms,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Virtual user coroutine
    # ------------------------------------------------------------------

    async def _virtual_user(self, session: aiohttp.ClientSession, user_id: int):
        """Coroutine representing one virtual user's lifecycle."""
        for _ in range(self.requests_per_user):
            req = self._build_request(user_id)
            result = await self._send_request(session, req)
            self.results.append(result)

            status_tag = "OK" if result.success else f"FAIL({result.error[:40]})"
            log.debug(
                "user=%d req=%s latency=%.1fms status=%s",
                user_id, req.request_id[:8], result.latency_ms, status_tag,
            )

            if self.think_time_ms > 0:
                await asyncio.sleep(self.think_time_ms / 1000)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self):
        """Spawn all virtual-user coroutines concurrently and await completion."""
        connector = aiohttp.TCPConnector(limit=0)           # no connection cap
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._virtual_user(session, uid)
                for uid in range(1, self.num_users + 1)
            ]
            log.info("Starting %d virtual users → %s", self.num_users, self.lb_url)
            wall_start = time.perf_counter()
            await asyncio.gather(*tasks)
            wall_time = time.perf_counter() - wall_start

        self._report(wall_time)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _report(self, wall_time: float):
        """Print a summary of the load-test results."""
        total = len(self.results)
        if total == 0:
            log.warning("No results collected.")
            return

        successes = [r for r in self.results if r.success]
        failures  = [r for r in self.results if not r.success]
        latencies = [r.latency_ms for r in successes]

        latencies.sort()
        p50  = latencies[int(len(latencies) * 0.50)] if latencies else 0
        p95  = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99  = latencies[int(len(latencies) * 0.99)] if latencies else 0
        avg  = sum(latencies) / len(latencies) if latencies else 0
        rps  = total / wall_time if wall_time > 0 else 0

        print("\n" + "=" * 60)
        print("           LOAD TEST REPORT")
        print("=" * 60)
        print(f"  Total Requests   : {total}")
        print(f"  Successful       : {len(successes)}")
        print(f"  Failed           : {len(failures)}")
        print(f"  Success Rate     : {100 * len(successes) / total:.1f}%")
        print(f"  Wall-clock Time  : {wall_time:.2f}s")
        print(f"  Throughput       : {rps:.1f} req/s")
        print(f"  Avg Latency      : {avg:.1f} ms")
        print(f"  P50 Latency      : {p50:.1f} ms")
        print(f"  P95 Latency      : {p95:.1f} ms")
        print(f"  P99 Latency      : {p99:.1f} ms")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Cluster — Client Load Generator")
    parser.add_argument("--users",    type=int, default=100,   help="Number of concurrent virtual users")
    parser.add_argument("--rpu",      type=int, default=1,     help="Requests per user")
    parser.add_argument("--think",    type=int, default=0,     help="Think time between requests (ms)")
    parser.add_argument("--timeout",  type=int, default=30,    help="Per-request timeout (s)")
    parser.add_argument("--lb-host",  type=str, default="127.0.0.1")
    parser.add_argument("--lb-port",  type=int, default=8080)
    return parser.parse_args()


def run_load_test(
    num_users: int = 100,
    requests_per_user: int = 1,
    lb_host: str = "127.0.0.1",
    lb_port: int = 8080,
):
    """Convenience wrapper called from main.py."""
    url = f"http://{lb_host}:{lb_port}/request"
    generator = LoadGenerator(
        lb_url=url,
        num_users=num_users,
        requests_per_user=requests_per_user,
    )
    asyncio.run(generator.run())


if __name__ == "__main__":
    args = parse_args()
    url = f"http://{args.lb_host}:{args.lb_port}/request"
    generator = LoadGenerator(
        lb_url=url,
        num_users=args.users,
        requests_per_user=args.rpu,
        think_time_ms=args.think,
        timeout_s=args.timeout,
    )
    asyncio.run(generator.run())
