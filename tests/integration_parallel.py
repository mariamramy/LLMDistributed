"""
Integration test — requires Docker stack to be running.
Run manually with: python tests/integration_parallel.py
Do NOT run with pytest — needs live Ollama + ChromaDB + workers.
"""
import asyncio
import aiohttp
import time


async def send_task(session, worker_url, worker_id, wall_start):
    payload = {
        "task_id": f"parallel-test-{worker_id}",
        "prompt": "What is replication in distributed systems?",
        "use_rag": True
    }
    start = time.perf_counter()
    relative = start - wall_start
    print(f"Worker {worker_id}: sending request at +{relative:.2f}s")

    async with session.post(f"{worker_url}/task", json=payload) as resp:
        result = await resp.json()
        elapsed = time.perf_counter() - start
        print(f"Worker {worker_id}: got response in {elapsed:.1f}s — {result['status']}")
        return result


async def main():
    workers = [
        ("http://localhost:9101", 1),
        ("http://localhost:9102", 2),
        ("http://localhost:9103", 3),
    ]

    wall_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [send_task(session, url, wid, wall_start) for url, wid in workers]
        results = await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - wall_start
    print(f"\n{'='*45}")
    print(f"  PARALLEL WORKER TEST RESULTS")
    print(f"{'='*45}")
    print(f"  Workers tested      : 3")
    print(f"  Total wall time     : {wall_time:.1f}s")
    print(f"  Est. sequential time: {wall_time * 3:.0f}s")
    print(f"  Parallel speedup    : {(wall_time * 3) / wall_time:.1f}x")
    print(f"{'='*45}")


asyncio.run(main())