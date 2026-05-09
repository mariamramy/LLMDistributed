"""
Integration load test — requires Docker stack to be running.
Run manually with: python tests/integration_load_test.py
Do NOT run with pytest — needs live Ollama + ChromaDB + workers.
"""
import asyncio
import aiohttp
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
]

LB_URL = "http://localhost:8080/request"

async def send_task(session, task_id, wall_start):
    payload = {
        "prompt": random.choice(PROMPTS),
        "use_rag": random.choice([True, False]),
    }
    start = time.perf_counter()
    try:
        async with session.post(
            LB_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=150)
        ) as resp:
            result = await resp.json()
            elapsed = time.perf_counter() - start
            status = result.get("status", "unknown")
            worker = result.get("worker_id", "?")
            rag = result.get("rag", {}).get("used", False)
            print(f"  Task {task_id:03d} → {worker} | {elapsed:.1f}s | {status} | rag={rag}")
            return {"success": status == "completed", "latency": elapsed, "worker": worker}
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  Task {task_id:03d} → FAILED | {elapsed:.1f}s | {e}")
        return {"success": False, "latency": elapsed, "worker": "none"}


async def main():
    NUM_REQUESTS = 30  # 3 per worker — change to 30 for a longer test

    print(f"\nSending {NUM_REQUESTS} requests through the full pipeline...")
    print(f"Route: Client → Load Balancer → Master → Workers → Ollama")
    print(f"{'='*60}")

    wall_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [send_task(session, i, wall_start) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - wall_start
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successes]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Count per worker
    worker_counts = {}
    for r in results:
        w = r["worker"]
        worker_counts[w] = worker_counts.get(w, 0) + 1

    print(f"\n{'='*60}")
    print(f"  LOAD TEST RESULTS — FULL PIPELINE")
    print(f"{'='*60}")
    print(f"  Total requests      : {NUM_REQUESTS}")
    print(f"  Successful          : {len(successes)}")
    print(f"  Failed              : {len(failures)}")
    print(f"  Success rate        : {100 * len(successes) / NUM_REQUESTS:.1f}%")
    print(f"  Wall-clock time     : {wall_time:.1f}s")
    print(f"  Avg latency         : {avg_latency:.1f}s")
    print(f"  Throughput          : {NUM_REQUESTS / wall_time:.3f} req/s")
    print(f"{'='*60}")
    print(f"  Requests per worker :")
    for worker, count in sorted(worker_counts.items()):
        print(f"    {worker}: {count} requests")
    print(f"{'='*60}")
    print(f"\n  Extrapolated for 1000 requests:")
    est_minutes = (1000 / NUM_REQUESTS) * wall_time / 60
    print(f"  Estimated time      : {est_minutes:.1f} minutes")
    print(f"{'='*60}\n")


asyncio.run(main())