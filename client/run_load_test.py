import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a parameterized Locust load test and print a summary report."
        )
    )
    parser.add_argument("--users", type=_positive_int, required=True)
    parser.add_argument(
        "--ramp-minutes",
        type=_positive_float,
        required=True,
        help="Minutes used to ramp up to the target users.",
    )
    parser.add_argument(
        "--run-minutes",
        type=_positive_float,
        default=5,
        help="Total load test duration in minutes.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="Target load balancer host.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=_positive_float,
        default=1.0,
        help="Per-user request interval in seconds.",
    )
    parser.add_argument(
        "--csv-prefix",
        default="results/load_test",
        help="CSV output prefix (without suffix like _stats.csv).",
    )
    parser.add_argument(
        "--locustfile",
        default="client/locustfile.py",
        help="Path to locustfile.",
    )
    return parser


def read_aggregate(stats_path: Path) -> tuple[float, int, int]:
    with stats_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Type") == "Aggregated":
                avg_response = float(row.get("Average Response Time", 0) or 0)
                requests = int(float(row.get("Request Count", 0) or 0))
                failures = int(float(row.get("Failure Count", 0) or 0))
                return avg_response, requests, failures
    raise RuntimeError(f"Aggregated stats not found in {stats_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    spawn_rate = args.users / (args.ramp_minutes * 60.0)
    if spawn_rate < 0.01:
        spawn_rate = 0.01

    csv_prefix = Path(args.csv_prefix)
    csv_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "locust",
        "-f",
        args.locustfile,
        "--host",
        args.host,
        "--headless",
        "-u",
        str(args.users),
        "-r",
        f"{spawn_rate:.4f}",
        "--run-time",
        f"{args.run_minutes}m",
        "--csv",
        str(csv_prefix),
    ]

    env = dict(**__import__("os").environ)
    env["LOCUST_WAIT_MIN"] = str(args.interval_seconds)
    env["LOCUST_WAIT_MAX"] = str(args.interval_seconds)

    print("Starting load test...")
    print(
        f"users={args.users}, ramp_minutes={args.ramp_minutes}, "
        f"spawn_rate={spawn_rate:.4f}/s, interval={args.interval_seconds}s"
    )

    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        return completed.returncode

    stats_path = Path(f"{args.csv_prefix}_stats.csv")
    avg_response, requests, failures = read_aggregate(stats_path)
    successes = requests - failures

    print("\nLoad Test Report")
    print(f"Average response time: {avg_response:.2f} ms")
    print(f"Successful requests: {successes}")
    print(f"Total requests: {requests}")
    print(f"Failed requests: {failures}")
    print(f"CSV output: {args.csv_prefix}_*.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
