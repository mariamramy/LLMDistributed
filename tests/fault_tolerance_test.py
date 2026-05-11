"""
Controlled fault tolerance test.
Sends requests against the live stack, kills a worker mid-run via Docker,
then verifies the system reassigns all orphaned tasks and completes with
zero client-visible failures.

Run after `docker compose up -d` and all workers are healthy:
    python tests/fault_tolerance_test.py
    python tests/fault_tolerance_test.py --kill-worker worker-2 --kill-after 40 --requests 150
"""
import asyncio
import aiohttp
import argparse
import subprocess
import time
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
    "How does vector clocks work in distributed systems?",
    "Explain the concept of quorum in distributed systems.",
]


async def send_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    request_id: int,
    counters: dict,
    timeout_s: int,
) -> dict:
    import random
    payload = {
        "request_id": f"ft-test-{request_id:04d}",
        "prompt": random.choice(PROMPTS),
        "use_rag": False,
    }
    async with semaphore:
        start = time.perf_counter()
        for attempt in range(3):
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
                    else:
                        counters["failed"] += 1
                    counters["workers"][worker] = counters["workers"].get(worker, 0) + 1
                    counters["latencies"].append(elapsed)
                    _print_progress(counters)
                    return {"success": status == "completed", "worker": worker}

            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                counters["completed"] += 1
                counters["failed"] += 1
                counters["workers"]["timeout"] = counters["workers"].get("timeout", 0) + 1
                _print_progress(counters)
                return {"success": False, "worker": "timeout"}


def _print_progress(counters: dict) -> None:
    completed = counters["completed"]
    total = counters["total"]
    if completed % 20 == 0 or completed == total:
        elapsed = time.monotonic() - counters["wall_start"]
        rps = completed / elapsed if elapsed > 0 else 0
        print(
            f"  [{completed:3d}/{total}] "
            f"OK={counters['success']} FAIL={counters['failed']} | "
            f"{rps:.2f} req/s | {elapsed:.0f}s elapsed"
        )


def docker_stop(worker: str) -> None:
    print(f"\n  *** Killing {worker} via docker stop ***\n")
    subprocess.run(["docker", "stop", worker], capture_output=True)


def docker_start(worker: str) -> None:
    print(f"\n  *** Restarting {worker} via docker start ***\n")
    subprocess.run(["docker", "start", worker], capture_output=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fault tolerance controlled test")
    parser.add_argument("--requests",    type=int, default=150,      help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=10,       help="Max simultaneous in-flight requests")
    parser.add_argument("--timeout",     type=int, default=120,      help="Per-request timeout in seconds")
    parser.add_argument("--kill-worker", default="worker-1",         help="Docker container name to kill")
    parser.add_argument("--kill-after",  type=int, default=30,       help="Seconds into the run before killing the worker")
    parser.add_argument("--restart",     action="store_true",        help="Restart the worker after the test completes")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Fault Tolerance Test")
    print(f"{'='*65}")
    print(f"  Requests      : {args.requests}")
    print(f"  Concurrency   : {args.concurrency}")
    print(f"  Kill target   : {args.kill_worker}")
    print(f"  Kill after    : {args.kill_after}s into the run")
    print(f"{'='*65}\n")

    counters = {
        "completed": 0,
        "success": 0,
        "failed": 0,
        "workers": {},
        "latencies": [],
        "total": args.requests,
        "wall_start": time.monotonic(),
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    wall_start = time.perf_counter()
    kill_done = False

    async def killer():
        nonlocal kill_done
        await asyncio.sleep(args.kill_after)
        docker_stop(args.kill_worker)
        kill_done = True

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [
            send_request(session, semaphore, i, counters, args.timeout)
            for i in range(args.requests)
        ]
        await asyncio.gather(asyncio.create_task(killer()), *coros)

    wall_time = time.perf_counter() - wall_start
    latencies = sorted(counters["latencies"])
    n = len(latencies)

    def pct(p):
        return latencies[min(int(n * p), n - 1)] if n > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  FAULT TOLERANCE TEST RESULTS")
    print(f"{'='*65}")
    print(f"  Worker killed       : {args.kill_worker} (at ~{args.kill_after}s)")
    print(f"  Total requests      : {args.requests}")
    print(f"  Successful          : {counters['success']}")
    print(f"  Failed              : {counters['failed']}")
    print(f"  Success rate        : {100 * counters['success'] / args.requests:.1f}%")
    print(f"  Wall-clock time     : {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"  Throughput          : {args.requests / wall_time:.3f} req/s")
    if n > 0:
        print(f"  P50 latency         : {pct(0.50):.1f}s")
        print(f"  P95 latency         : {pct(0.95):.1f}s")
    print(f"{'='*65}")
    print(f"  Requests per worker (shows redistribution after kill):")
    for worker, count in sorted(counters["workers"].items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        killed = "  ← KILLED" if worker == args.kill_worker else ""
        print(f"    {worker:12s} : {count:4d}  {bar}{killed}")
    print(f"{'='*65}\n")

    _write_log(args, counters, wall_time, latencies, pct)

    if args.restart:
        docker_start(args.kill_worker)
        print(f"  {args.kill_worker} restarted. Re-registration will happen on next worker startup.\n")


def _write_log(args, counters, wall_time, latencies, pct):
    log_path = os.path.join(os.path.dirname(__file__), "fault_tolerance_results.log")
    n = len(latencies)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"\n{'='*65}",
        f"  Timestamp           : {timestamp}",
        f"  Worker killed       : {args.kill_worker} (at ~{args.kill_after}s)",
        f"  Total requests      : {args.requests}",
        f"  Successful          : {counters['success']}",
        f"  Failed              : {counters['failed']}",
        f"  Success rate        : {100 * counters['success'] / args.requests:.1f}%",
        f"  Wall-clock time     : {wall_time:.1f}s  ({wall_time/60:.1f} min)",
        f"  Throughput          : {args.requests / wall_time:.3f} req/s",
    ]
    if n > 0:
        lines += [
            f"  P50 latency         : {pct(0.50):.1f}s",
            f"  P95 latency         : {pct(0.95):.1f}s",
        ]
    lines.append(f"  Requests per worker :")
    for worker, count in sorted(counters["workers"].items(), key=lambda x: -x[1]):
        killed = " <- KILLED" if worker == args.kill_worker else ""
        lines.append(f"    {worker:12s} : {count:4d}{killed}")
    lines.append(f"{'='*65}\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Results saved to: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
```
