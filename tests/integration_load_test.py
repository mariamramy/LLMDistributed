"""
Integration load test - requires Docker stack to be running.
Run manually with: python tests/integration_load_test.py [--requests N] [--concurrency C]
Do NOT run with pytest - needs live Ollama + ChromaDB + workers.

Full pipeline: Client → LB (8080) → Master (9000) → Workers → Ollama
"""
import asyncio
import aiohttp
import argparse
import subprocess
import time
import random
import datetime
import os

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
    parser.add_argument("--requests",    type=int,            default=1000, help="Total requests to send")
    parser.add_argument("--concurrency", type=int,            default=10,   help="Max simultaneous in-flight requests")
    parser.add_argument("--timeout",     type=int,            default=300,  help="Per-request timeout in seconds")
    parser.add_argument("--monitor-gpu", action="store_true",               help="Capture GPU stats in parallel via gpu_monitor.py")
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

    # Launch GPU monitor as a background process if requested
    gpu_proc = None
    if args.monitor_gpu:
        monitor_script = os.path.join(os.path.dirname(__file__), "gpu_monitor.py")
        gpu_proc = subprocess.Popen(
            ["python", monitor_script, "--duration", str(TIMEOUT_S + 120), "--interval", "5"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"  GPU monitor started (pid={gpu_proc.pid})\n")

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
        print(f"\n  Estimated time for 1000 requests : ~{est_1000_min:.1f} min")
    print(f"{'='*65}\n")

    # Stop GPU monitor
    if gpu_proc:
        gpu_proc.terminate()
        gpu_proc.wait()
        print(f"  GPU monitor stopped — stats saved to tests/results/")

    _write_results_log(NUM_REQUESTS, MAX_CONCURRENCY, TIMEOUT_S,
                       counters, wall_time, latencies, pct, est_1000_min if n > 0 else None)


def _write_results_log(num_req, concurrency, timeout_s, counters, wall_time, latencies, pct, est_1000_min):
    import json
    tests_dir  = os.path.dirname(__file__)
    results_dir = os.path.join(tests_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    n = len(latencies)
    timestamp     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_fn  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    throughput    = num_req / wall_time
    success_rate  = 100 * counters["success"] / num_req

    lines = [
        f"\n{'='*65}",
        f"  Run timestamp       : {timestamp}",
        f"  Total requests      : {num_req}",
        f"  Concurrency         : {concurrency}",
        f"  Timeout             : {timeout_s}s",
        f"{'='*65}",
        f"  Successful          : {counters['success']}",
        f"  Failed              : {counters['failed']}",
        f"  Success rate        : {success_rate:.1f}%",
        f"  Wall-clock time     : {wall_time:.1f}s  ({wall_time / 60:.1f} min)",
        f"  Throughput          : {throughput:.3f} req/s",
    ]
    if n > 0:
        avg = sum(latencies) / n
        lines += [
            f"  Avg latency         : {avg:.1f}s",
            f"  P50 latency         : {pct(0.50):.1f}s",
            f"  P95 latency         : {pct(0.95):.1f}s",
            f"  P99 latency         : {pct(0.99):.1f}s",
            f"  Min latency         : {latencies[0]:.1f}s",
            f"  Max latency         : {latencies[-1]:.1f}s",
        ]
    lines.append(f"{'='*65}")
    lines.append(f"  Requests per worker :")
    for worker, count in sorted(counters["workers"].items(), key=lambda x: -x[1]):
        lines.append(f"    {worker:12s} : {count:4d}")
    if est_1000_min is not None:
        lines.append(f"\n  Estimated time for 1000 requests : ~{est_1000_min:.1f} min")
    lines.append(f"{'='*65}\n")
    content = "\n".join(lines) + "\n"

    # 1. Cumulative log (all runs appended)
    cumulative_path = os.path.join(tests_dir, "load_test_results.log")
    with open(cumulative_path, "a", encoding="utf-8") as f:
        f.write(content)

    # 2. Individual run file (for screenshots)
    individual_path = os.path.join(results_dir, f"load_test_{num_req}req_{timestamp_fn}.txt")
    with open(individual_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 3. Graph data JSON (accumulates key metrics across runs)
    graph_path = os.path.join(results_dir, "graph_data.json")
    graph_data = []
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            try:
                graph_data = json.load(f)
            except Exception:
                graph_data = []
    graph_data.append({
        "timestamp":    timestamp,
        "num_requests": num_req,
        "concurrency":  concurrency,
        "success_rate": round(success_rate, 1),
        "throughput":   round(throughput, 3),
        "wall_time_s":  round(wall_time, 1),
        "p50_s":        round(pct(0.50), 1) if n > 0 else None,
        "p95_s":        round(pct(0.95), 1) if n > 0 else None,
        "p99_s":        round(pct(0.99), 1) if n > 0 else None,
        "min_s":        round(latencies[0], 1) if n > 0 else None,
        "max_s":        round(latencies[-1], 1) if n > 0 else None,
        "avg_s":        round(sum(latencies)/n, 1) if n > 0 else None,
        "workers":      dict(sorted(counters["workers"].items(), key=lambda x: -x[1])),
    })
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"  Cumulative log : {cumulative_path}")
    print(f"  This run log   : {individual_path}")
    print(f"  Graph data     : {graph_path}")


if __name__ == "__main__":
    asyncio.run(main())
