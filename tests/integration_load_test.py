"""
Integration load test - requires Docker stack to be running.
Run manually with: python tests/integration_load_test.py [--requests N] [--concurrency C]
Do NOT run with pytest - needs live Ollama + ChromaDB + workers.

Full pipeline: Client → LB (8080) → Master (9000) → Workers → Ollama
"""
import asyncio
import aiohttp
import argparse
import time
import random

PROMPTS = [
    "What is replication in distributed systems?",
    "Explain fault tolerance in distributed systems.",
    "What is the CAP theorem?",
    "How does consensus work in distributed systems?",
    "What is a distributed hash table?",
    "Explain the difference between consistency and availability.",
    "What is a leader election algorithm?",
    "How does Paxos work?",
    "What is eventual consistency?",
    "Explain sharding in distributed databases.",
    "What is two-phase commit?",
    "Describe the Raft consensus algorithm.",
    "How does vector clocks work in distributed systems?",
    "What is the difference between strong and eventual consistency?",
    "Explain the concept of quorum in distributed systems.",
]

LB_URL = "http://localhost:8080/request"


async def send_request(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                       request_id: int, counters: dict, timeout_s: int,
                       max_retries: int = 2) -> dict:
    payload = {
        "request_id": f"load-test-{request_id:05d}",
        "prompt": random.choice(PROMPTS),
        "use_rag": random.choice([True, False]),
    }
    start = time.perf_counter()
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                async with session.post(
                    LB_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as resp:
                    result = await resp.json()
                    elapsed = time.perf_counter() - start
                    status = result.get("status", "unknown")
                    worker = result.get("worker_id", "?")

                    if status == "completed":
                        counters["completed"] += 1
                        counters["success"] += 1
                        counters["latencies"].append(elapsed)
                        counters["workers"][worker] = counters["workers"].get(worker, 0) + 1
                        _print_progress(counters)
                        return {"success": True, "latency": elapsed, "worker": worker}
                    elif attempt < max_retries:
                        await asyncio.sleep(2)
                        continue
                    else:
                        counters["completed"] += 1
                        counters["failed"] += 1
                        _print_progress(counters)
                        return {"success": False, "latency": elapsed, "worker": worker}

            except (asyncio.TimeoutError, Exception):
                if attempt < max_retries:
                    await asyncio.sleep(2)
                    continue
                elapsed = time.perf_counter() - start
                counters["completed"] += 1
                counters["failed"] += 1
                _print_progress(counters)
                return {"success": False, "latency": elapsed, "worker": "timeout"}


def _print_progress(counters: dict) -> None:
    completed = counters["completed"]
    total = counters["total"]
    if completed % 50 == 0 or completed == total:
        pct = 100 * completed / total
        elapsed = time.monotonic() - counters["wall_start"]
        rps = completed / elapsed if elapsed > 0 else 0
        print(
            f"  [{completed:4d}/{total}] {pct:5.1f}% | "
            f"OK={counters['success']} FAIL={counters['failed']} | "
            f"{rps:.2f} req/s | {elapsed:.0f}s elapsed"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM cluster integration load test")
    parser.add_argument("--requests",    type=int, default=1000, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=10,   help="Max simultaneous in-flight requests")
    parser.add_argument("--timeout",     type=int, default=300,  help="Per-request timeout in seconds")
    args = parser.parse_args()

    NUM_REQUESTS = args.requests
    MAX_CONCURRENCY = args.concurrency
    TIMEOUT_S = args.timeout

    print(f"\n{'='*65}")
    print(f"  LLM Distributed System - Integration Load Test")
    print(f"{'='*65}")
    print(f"  Requests    : {NUM_REQUESTS}")
    print(f"  Concurrency : {MAX_CONCURRENCY} simultaneous in-flight")
    print(f"  Timeout     : {TIMEOUT_S}s per request")
    print(f"  Route       : Client -> LB:8080 -> Master:9000 -> Workers -> Ollama")
    print(f"{'='*65}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    counters: dict = {
        "completed": 0,
        "success": 0,
        "failed": 0,
        "latencies": [],
        "workers": {},
        "total": NUM_REQUESTS,
        "wall_start": time.monotonic(),
    }

    wall_start = time.perf_counter()

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            send_request(session, semaphore, i, counters, TIMEOUT_S)
            for i in range(NUM_REQUESTS)
        ]
        await asyncio.gather(*coros)

    wall_time = time.perf_counter() - wall_start
    latencies = sorted(counters["latencies"])
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(n * p), n - 1)] if n > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  LOAD TEST RESULTS - FULL PIPELINE")
    print(f"{'='*65}")
    print(f"  Total requests      : {NUM_REQUESTS}")
    print(f"  Successful          : {counters['success']}")
    print(f"  Failed              : {counters['failed']}")
    print(f"  Success rate        : {100 * counters['success'] / NUM_REQUESTS:.1f}%")
    print(f"  Wall-clock time     : {wall_time:.1f}s  ({wall_time / 60:.1f} min)")
    print(f"  Throughput          : {NUM_REQUESTS / wall_time:.3f} req/s")
    if n > 0:
        avg = sum(latencies) / n
        print(f"  Avg latency         : {avg:.1f}s")
        print(f"  P50 latency         : {pct(0.50):.1f}s")
        print(f"  P95 latency         : {pct(0.95):.1f}s")
        print(f"  P99 latency         : {pct(0.99):.1f}s")
        print(f"  Min latency         : {latencies[0]:.1f}s")
        print(f"  Max latency         : {latencies[-1]:.1f}s")
    print(f"{'='*65}")
    print(f"  Requests per worker :")
    for worker, count in sorted(counters["workers"].items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        print(f"    {worker:12s} : {count:4d}  {bar}")
    print(f"{'='*65}")
    if n > 0:
        est_1000_min = (1000 / NUM_REQUESTS) * wall_time / 60
        print(f"\n  Extrapolated for 1000 requests : ~{est_1000_min:.1f} min")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
