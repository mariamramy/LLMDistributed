"""
Stub load test — measures pure distributed system performance with no LLM overhead.
Workers must be running with STUB_MODE=true (set via docker compose).

This isolates: LB routing, master scheduling, queue management, worker concurrency.
Compare results against integration_load_test.py to quantify LLM inference overhead.

Run:
    # 1. Restart workers in stub mode:
    #    docker compose up -d --build  (after adding STUB_MODE: "true" to docker-compose.yml)
    # 2. Run this test:
    python tests/stub_load_test.py --requests 1000 --concurrency 50
"""
import asyncio
import aiohttp
import argparse
import time
import random
import datetime
import os

LB_URL = "http://localhost:8080/request"

PROMPTS = [
    "What is the CAP theorem?",
    "Explain fault tolerance in distributed systems.",
    "What is a leader election algorithm?",
    "How does Paxos work?",
    "What is eventual consistency?",
    "Explain sharding in distributed databases.",
    "What is two-phase commit?",
    "Describe the Raft consensus algorithm.",
]


async def send_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    request_id: int,
    counters: dict,
    timeout_s: int,
) -> None:
    payload = {
        "request_id": f"stub-{request_id:05d}",
        "prompt": random.choice(PROMPTS),
        "use_rag": False,
    }
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(
                LB_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                result = await resp.json()
                elapsed = time.perf_counter() - start
                worker = result.get("worker_id", "unknown")
                status = result.get("status", "unknown")

                counters["completed"] += 1
                if status == "completed":
                    counters["success"] += 1
                    counters["latencies"].append(elapsed)
                    counters["workers"][worker] = counters["workers"].get(worker, 0) + 1
                else:
                    counters["failed"] += 1
                _print_progress(counters)

        except Exception:
            elapsed = time.perf_counter() - start
            counters["completed"] += 1
            counters["failed"] += 1
            counters["latencies"].append(elapsed)
            _print_progress(counters)


def _print_progress(counters: dict) -> None:
    completed = counters["completed"]
    total = counters["total"]
    if completed % 100 == 0 or completed == total:
        elapsed = time.monotonic() - counters["wall_start"]
        rps = completed / elapsed if elapsed > 0 else 0
        print(
            f"  [{completed:4d}/{total}] {100*completed/total:5.1f}% | "
            f"OK={counters['success']} FAIL={counters['failed']} | "
            f"{rps:.1f} req/s | {elapsed:.1f}s elapsed"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stub load test — no LLM inference")
    parser.add_argument("--requests",    type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout",     type=int, default=30)
    args = parser.parse_args()

    # Verify workers are in stub mode
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("http://localhost:9101/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                health = await r.json()
                if not health.get("stub_mode"):
                    print("\n  WARNING: worker-1 does not appear to be in stub mode.")
                    print("  Add STUB_MODE: 'true' to each worker in docker-compose.yml and rebuild.\n")
    except Exception:
        print("  WARNING: Could not reach worker-1 health endpoint.\n")

    print(f"\n{'='*65}")
    print(f"  Stub Load Test — Pure System Performance (no LLM)")
    print(f"{'='*65}")
    print(f"  Requests    : {args.requests}")
    print(f"  Concurrency : {args.concurrency} simultaneous in-flight")
    print(f"  Timeout     : {args.timeout}s per request")
    print(f"  Route       : Client -> LB:8080 -> Master:9000 -> Workers (stub)")
    print(f"{'='*65}\n")

    counters = {
        "completed": 0, "success": 0, "failed": 0,
        "latencies": [], "workers": {},
        "total": args.requests,
        "wall_start": time.monotonic(),
    }

    wall_start = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            send_request(session, semaphore, i, counters, args.timeout)
            for i in range(args.requests)
        ]
        await asyncio.gather(*coros)

    wall_time = time.perf_counter() - wall_start
    latencies = sorted(counters["latencies"])
    n = len(latencies)

    def pct(p):
        return latencies[min(int(n * p), n - 1)] if n > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  STUB LOAD TEST RESULTS")
    print(f"{'='*65}")
    print(f"  Total requests      : {args.requests}")
    print(f"  Successful          : {counters['success']}")
    print(f"  Failed              : {counters['failed']}")
    print(f"  Success rate        : {100 * counters['success'] / args.requests:.1f}%")
    print(f"  Wall-clock time     : {wall_time:.2f}s  ({wall_time/60:.2f} min)")
    print(f"  Throughput          : {args.requests / wall_time:.1f} req/s")
    if n > 0:
        print(f"  Avg latency         : {sum(latencies)/n*1000:.1f}ms")
        print(f"  P50 latency         : {pct(0.50)*1000:.1f}ms")
        print(f"  P95 latency         : {pct(0.95)*1000:.1f}ms")
        print(f"  P99 latency         : {pct(0.99)*1000:.1f}ms")
        print(f"  Min latency         : {latencies[0]*1000:.1f}ms")
        print(f"  Max latency         : {latencies[-1]*1000:.1f}ms")
    print(f"{'='*65}")
    print(f"  Requests per worker :")
    for worker, count in sorted(counters["workers"].items(), key=lambda x: -x[1]):
        bar = "#" * min(count // 5, 40)
        print(f"    {worker:12s} : {count:4d}  {bar}")
    print(f"{'='*65}\n")

    _write_log(args, counters, wall_time, latencies, pct, n)


def _write_log(args, counters, wall_time, latencies, pct, n):
    log_path = os.path.join(os.path.dirname(__file__), "results", "stub_load_test_results.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"\n{'='*65}",
        f"  Timestamp           : {timestamp}",
        f"  Mode                : STUB (no LLM inference)",
        f"  Total requests      : {args.requests}",
        f"  Concurrency         : {args.concurrency}",
        f"  Successful          : {counters['success']}",
        f"  Failed              : {counters['failed']}",
        f"  Success rate        : {100 * counters['success'] / args.requests:.1f}%",
        f"  Wall-clock time     : {wall_time:.2f}s",
        f"  Throughput          : {args.requests / wall_time:.1f} req/s",
    ]
    if n > 0:
        lines += [
            f"  P50 latency         : {pct(0.50)*1000:.1f}ms",
            f"  P95 latency         : {pct(0.95)*1000:.1f}ms",
            f"  P99 latency         : {pct(0.99)*1000:.1f}ms",
        ]
    lines.append(f"  Requests per worker :")
    for worker, count in sorted(counters["workers"].items(), key=lambda x: -x[1]):
        lines.append(f"    {worker:12s} : {count:4d}")
    lines.append(f"{'='*65}\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Results saved to: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
