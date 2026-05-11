"""
GPU monitoring script — run alongside any load test to capture GPU stats.

Usage (in a separate terminal while load test is running):
    python tests/gpu_monitor.py --duration 900 --interval 5

Or launch automatically from the load test with --monitor-gpu flag.

Output saved to: tests/results/gpu_stats_YYYY-MM-DD_HH-MM.txt
"""
import subprocess
import time
import os
import datetime
import argparse
import sys


def collect_sample() -> dict:
    """Run nvidia-smi and parse memory + temperature + power."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,"
                "temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "index":      parts[0],
            "name":       parts[1],
            "mem_used":   parts[2],
            "mem_total":  parts[3],
            "mem_free":   parts[4],
            "temp_c":     parts[5],
            "power_w":    parts[6],
            "util_pct":   parts[7],
        }
    except Exception:
        return None


def format_sample(ts: str, s: dict) -> str:
    util = f"{s['util_pct']}%" if s["util_pct"] not in ("[Unknown Error]", "") else "N/A"
    return (
        f"  {ts}  |  "
        f"VRAM: {s['mem_used']} / {s['mem_total']} MiB  "
        f"({s['mem_free']} MiB free)  |  "
        f"Temp: {s['temp_c']}°C  |  "
        f"Power: {s['power_w']}W  |  "
        f"GPU util: {util}"
    )


def run(duration_s: int, interval_s: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start = time.monotonic()
    samples = []

    print(f"\n  GPU Monitor started — collecting every {interval_s}s for {duration_s}s")
    print(f"  Output: {output_path}\n")

    header_sample = collect_sample()
    gpu_name = header_sample["name"] if header_sample else "Unknown GPU"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*75}\n")
        f.write(f"  GPU Monitor Log\n")
        f.write(f"  GPU        : {gpu_name}\n")
        f.write(f"  Started    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Interval   : {interval_s}s\n")
        f.write(f"{'='*75}\n\n")

        while time.monotonic() - start < duration_s:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            sample = collect_sample()
            if sample:
                line = format_sample(ts, sample)
                print(f"  {line}")
                f.write(line + "\n")
                f.flush()
                samples.append(sample)
            else:
                msg = f"  {ts}  |  nvidia-smi unavailable"
                print(msg)
                f.write(msg + "\n")
                f.flush()
            time.sleep(interval_s)

        # Summary
        if samples:
            try:
                mem_values  = [int(s["mem_used"])  for s in samples if s["mem_used"].isdigit()]
                temp_values = [int(s["temp_c"])    for s in samples if s["temp_c"].isdigit()]
                pow_values  = [float(s["power_w"]) for s in samples]

                f.write(f"\n{'='*75}\n")
                f.write(f"  SUMMARY\n")
                f.write(f"{'='*75}\n")
                f.write(f"  Samples collected  : {len(samples)}\n")
                if mem_values:
                    f.write(f"  VRAM used  — avg: {sum(mem_values)//len(mem_values)} MiB  "
                            f"max: {max(mem_values)} MiB  "
                            f"min: {min(mem_values)} MiB\n")
                if temp_values:
                    f.write(f"  Temperature — avg: {sum(temp_values)//len(temp_values)}°C  "
                            f"max: {max(temp_values)}°C\n")
                if pow_values:
                    f.write(f"  Power draw  — avg: {sum(pow_values)/len(pow_values):.1f}W  "
                            f"max: {max(pow_values):.1f}W\n")
                f.write(f"  VRAM total         : {samples[0]['mem_total']} MiB\n")
                f.write(f"{'='*75}\n")
            except Exception:
                pass

    print(f"\n  GPU Monitor complete. Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU monitoring alongside load tests")
    parser.add_argument("--duration", type=int, default=300, help="How long to monitor in seconds")
    parser.add_argument("--interval", type=int, default=5,   help="Seconds between samples")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out = os.path.join(os.path.dirname(__file__), "results", f"gpu_stats_{ts}.txt")
    run(args.duration, args.interval, out)


if __name__ == "__main__":
    main()
