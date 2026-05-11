"""
Runs the controlled fault tolerance test and generates a visual timeline
showing the kill event and automatic task redistribution.

Run after `docker compose up -d`:
    python tests/fault_tolerance_timeline.py

Output: tests/charts/fault_tolerance_timeline.png
"""
import asyncio
import aiohttp
import argparse
import subprocess
import time
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

LB_URL = "http://localhost:8080/request"
PROMPTS = [
    "What is the CAP theorem?",
    "Explain fault tolerance in distributed systems.",
    "What is a leader election algorithm?",
    "How does Paxos work?",
    "What is eventual consistency?",
]

# Records per completed request
events: list = []
kill_time: float = None
wall_start: float = None


async def send_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    request_id: int,
    timeout_s: int,
) -> None:
    payload = {
        "request_id": f"ft-{request_id:04d}",
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
                    completed_at = time.perf_counter() - wall_start
                    events.append({
                        "request_id": request_id,
                        "worker": worker,
                        "completed_at": completed_at,
                        "latency": elapsed,
                        "success": status == "completed",
                    })
                    return
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                completed_at = time.perf_counter() - wall_start
                events.append({
                    "request_id": request_id,
                    "worker": "timeout",
                    "completed_at": completed_at,
                    "latency": time.perf_counter() - start,
                    "success": False,
                })
                return


async def killer(kill_after: float, kill_worker: str) -> None:
    global kill_time
    await asyncio.sleep(kill_after)
    kill_time = time.perf_counter() - wall_start
    print(f"\n  *** Killing {kill_worker} at t={kill_time:.1f}s ***\n")
    subprocess.run(["docker", "stop", kill_worker], capture_output=True)


async def run_test(requests: int, concurrency: int, timeout_s: int,
                   kill_worker: str, kill_after: float) -> None:
    global wall_start
    wall_start = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [send_request(session, semaphore, i, timeout_s)
                 for i in range(requests)]
        await asyncio.gather(asyncio.create_task(killer(kill_after, kill_worker)), *coros)


def generate_timeline(kill_worker: str, kill_after: float, restart: bool) -> None:
    all_workers = sorted(set(e["worker"] for e in events if e["worker"] != "timeout"))
    worker_colors = {
        "worker-1": "#4C72B0",
        "worker-2": "#55A868",
        "worker-3": "#DD8452",
        "worker-4": "#C44E52",
        "worker-5": "#8172B2",
    }
    default_color = "#999999"

    fig, ax = plt.subplots(figsize=(12, 5))

    for event in events:
        w = event["worker"]
        if w == "timeout" or not event["success"]:
            continue
        y = all_workers.index(w) if w in all_workers else -1
        color = worker_colors.get(w, default_color)
        killed = (w == kill_worker)
        marker = "x" if killed else "o"
        ax.scatter(event["completed_at"], y,
                   color=color, marker=marker,
                   s=35, alpha=0.7, linewidths=1.5 if killed else 0)

    # Kill line
    if kill_time is not None:
        ax.axvline(x=kill_time, color="red", linewidth=2, linestyle="--", zorder=5)
        ax.text(kill_time + 1, len(all_workers) - 0.4,
                f"  {kill_worker} killed\n  t={kill_time:.0f}s",
                color="red", fontsize=9, va="top")

    # Shading — before and after kill
    if kill_time is not None:
        ax.axvspan(0, kill_time, alpha=0.05, color="blue", label="Before kill")
        ax.axvspan(kill_time, max(e["completed_at"] for e in events) + 2,
                   alpha=0.05, color="orange", label="After kill — auto-recovery")

    ax.set_yticks(range(len(all_workers)))
    ax.set_yticklabels(all_workers, fontsize=11)
    ax.set_xlabel("Elapsed time (seconds)", fontsize=11)
    ax.set_title(
        f"Fault Tolerance Timeline — {kill_worker} killed at t={kill_time:.0f}s\n"
        f"System automatically redistributed tasks to healthy workers",
        fontsize=12, fontweight="bold"
    )

    # Stats
    before_kill = [e for e in events if e["completed_at"] <= kill_time and e["success"]]
    after_kill  = [e for e in events if e["completed_at"] > kill_time  and e["success"]]
    success_total = sum(1 for e in events if e["success"])
    total = len(events)

    from collections import Counter
    before_dist = Counter(e["worker"] for e in before_kill)
    after_dist  = Counter(e["worker"] for e in after_kill)

    stats_text = (
        f"Total: {total}  |  Success: {success_total} ({100*success_total/total:.0f}%)  |  "
        f"Failed: {total - success_total}\n"
        f"Before kill — {kill_worker}: {before_dist.get(kill_worker, 0)} req  |  "
        f"After kill — {kill_worker}: {after_dist.get(kill_worker, 0)} req"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    legend_handles = [
        mpatches.Patch(color=worker_colors.get(w, default_color), label=w)
        for w in all_workers
    ]
    legend_handles += [
        plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Kill event"),
        mpatches.Patch(color="blue", alpha=0.15, label="Before kill"),
        mpatches.Patch(color="orange", alpha=0.15, label="After kill (recovery)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, ncol=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(OUT_DIR, "fault_tolerance_timeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nTimeline saved: {path}")

    # Print summary
    print(f"\n{'='*55}")
    print(f"  FAULT TOLERANCE SUMMARY")
    print(f"{'='*55}")
    print(f"  Worker killed       : {kill_worker} at t={kill_time:.1f}s")
    print(f"  Total requests      : {len(events)}")
    print(f"  Successful          : {success_total}")
    print(f"  Failed              : {len(events) - success_total}")
    print(f"  Success rate        : {100*success_total/len(events):.1f}%")
    print(f"\n  Requests per worker :")
    all_counts = Counter(e["worker"] for e in events if e["success"])
    for w, c in sorted(all_counts.items(), key=lambda x: -x[1]):
        tag = "  <- KILLED" if w == kill_worker else ""
        print(f"    {w:12s} : {c:3d}{tag}")
    print(f"{'='*55}\n")

    # Save terminal summary to individual results file for screenshots
    import datetime
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = os.path.join(results_dir, f"fault_tolerance_{ts}.txt")
    lines = [
        f"{'='*55}",
        f"  FAULT TOLERANCE TEST",
        f"{'='*55}",
        f"  Timestamp           : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Worker killed       : {kill_worker} at t={kill_time:.1f}s",
        f"  Total requests      : {total}",
        f"  Successful          : {success_total}",
        f"  Failed              : {total - success_total}",
        f"  Success rate        : {100*success_total/total:.1f}%",
        f"  Outage window       : t={kill_time:.0f}s until Docker auto-restart",
        f"{'='*55}",
        f"  Requests per worker :",
    ]
    for w, c in sorted(all_counts.items(), key=lambda x: -x[1]):
        tag = "  <- KILLED" if w == kill_worker else ""
        lines.append(f"    {w:12s} : {c:3d}{tag}")
    lines += [
        f"{'='*55}",
        f"  Before kill — {kill_worker}: {before_dist.get(kill_worker, 0)} requests",
        f"  After kill  — {kill_worker}: {after_dist.get(kill_worker, 0)} requests (post-restart)",
        f"  Chart saved : {os.path.join(OUT_DIR, 'fault_tolerance_timeline.png')}",
        f"{'='*55}",
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Results saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests",    type=int,   default=120)
    parser.add_argument("--concurrency", type=int,   default=10)
    parser.add_argument("--timeout",     type=int,   default=120)
    parser.add_argument("--kill-worker", default="worker-1")
    parser.add_argument("--kill-after",  type=float, default=30)
    parser.add_argument("--restart",     action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Fault Tolerance Timeline Test")
    print(f"{'='*55}")
    print(f"  Requests    : {args.requests}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Kill target : {args.kill_worker} at {args.kill_after}s")
    print(f"{'='*55}\n")

    asyncio.run(run_test(
        args.requests, args.concurrency, args.timeout,
        args.kill_worker, args.kill_after,
    ))

    generate_timeline(args.kill_worker, args.kill_after, args.restart)

    if args.restart:
        print(f"Restarting {args.kill_worker}...")
        subprocess.run(["docker", "start", args.kill_worker], capture_output=True)
        print(f"{args.kill_worker} restarted.")


if __name__ == "__main__":
    main()
