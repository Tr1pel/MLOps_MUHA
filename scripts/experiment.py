"""Measurement harness for the backpressure experiment (README, "Эксперимент").

One run is one scenario: 1 worker at t=0, a second one at --scale-at, the source stopped at
--duration. The queue is polled through the RabbitMQ Management API once a second; this is only
*measurement*: the source never uses the API and gets its backpressure from publisher confirms.

    uv run python scripts/experiment.py run --label A --queue tasks-unbounded --max-length 0 --batch-id exp-a
    uv run python scripts/experiment.py cleanup --queue tasks-unbounded
    uv run python scripts/experiment.py run --label B --queue tasks --max-length 50 --batch-id exp-b

--queue and --max-length are exported as TASKS_QUEUE / QUEUE_MAX_LENGTH, so the app, the worker
and the source started from here get identical values (a queue declared with different
arguments is refused by RabbitMQ).
"""

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "backpressure"
SOURCE_CONTAINER = "exp-source"
PROGRESS = re.compile(
    r"source_progress published_total=(\d+) rejected_total=(\d+) rate_per_s=([\d.]+)"
)
UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return {**values, **{k: v for k, v in os.environ.items() if k in values}}


ENV = read_env()


def api(path: str, method: str = "GET") -> dict | list | None:
    token = base64.b64encode(f"{ENV['RABBITMQ_USER']}:{ENV['RABBITMQ_PASSWORD']}".encode()).decode()
    url = f"http://127.0.0.1:{ENV['RABBITMQ_MANAGEMENT_PORT']}/api{path}"
    request = Request(url, method=method, headers={"Authorization": f"Basic {token}"})
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return json.loads(body) if body else None


def compose(*args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
        **kwargs,
    )


def delete_queues(queue: str) -> None:
    for name in (queue, f"{queue}.dlq"):
        api(f"/queues/%2F/{quote(name)}", "DELETE")
    api(f"/exchanges/%2F/{quote(queue + '.dlx')}", "DELETE")


def restart_pipeline() -> None:
    """Recreate app and one worker with the current environment (queue arguments must agree)."""
    compose(
        "up",
        "-d",
        "--wait",
        "--no-deps",
        "--force-recreate",
        "--scale",
        "worker=1",
        "app",
        "worker",
    )


def prepare(queue: str) -> None:
    compose("stop", "worker")
    delete_queues(queue)
    restart_pipeline()


def docker_stats_loop(stop: threading.Event, t0: float, rows: list[dict]) -> None:
    while not stop.is_set():
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        now = time.time() - t0
        for line in out.splitlines():
            item = json.loads(line)
            name = item["Name"]
            if not any(part in name for part in ("rabbitmq", "worker", SOURCE_CONTAINER)):
                continue
            used = item["MemUsage"].split("/")[0].strip()
            number, unit = re.match(r"([\d.]+)\s*(\w+)", used).groups()
            rows.append(
                {
                    "t": round(now, 1),
                    "container": name,
                    "mem_bytes": int(float(number) * UNITS[unit]),
                    "cpu_percent": float(item["CPUPerc"].rstrip("%")),
                }
            )


def sample(t0: float) -> dict:
    queue = api(f"/queues/%2F/{quote(ENV['TASKS_QUEUE'])}") or {}
    node = (api("/nodes") or [{}])[0]
    stats = queue.get("message_stats", {})
    return {
        "t": round(time.time() - t0, 2),
        "messages_ready": queue.get("messages_ready", 0),
        "messages_unacknowledged": queue.get("messages_unacknowledged", 0),
        "consumers": queue.get("consumers", 0),
        "queue_memory_bytes": queue.get("memory", 0),
        "node_mem_used_bytes": node.get("mem_used", 0),
        "mgmt_publish_rate": stats.get("publish_details", {}).get("rate", 0),
        "mgmt_ack_rate": stats.get("ack_details", {}).get("rate", 0),
    }


def source_progress(t0: float) -> list[dict]:
    logs = subprocess.run(
        ["docker", "logs", "--timestamps", SOURCE_CONTAINER],
        capture_output=True,
        text=True,
        check=False,
    )
    rows, previous = [], (0, 0, 0.0)
    for line in (logs.stdout + logs.stderr).splitlines():
        match = PROGRESS.search(line)
        if not match:
            continue
        stamp = line.split(" ", 1)[0]
        moment = datetime.fromisoformat(stamp[:26].rstrip("Z") + "+00:00").timestamp()
        published, rejected, rate = int(match[1]), int(match[2]), float(match[3])
        span = max(moment - previous[2], 1e-9) if previous[2] else 1.0
        rows.append(
            {
                "t": round(moment - t0, 2),
                "published_total": published,
                "rejected_total": rejected,
                "published_per_s": rate,
                "rejected_per_s": round((rejected - previous[1]) / span, 1),
            }
        )
        previous = (published, rejected, moment)
    return rows


def task_times(t0: float, t_end: float) -> list[dict]:
    query = (
        "SELECT extract(epoch from finished_at), extract(epoch from enqueued_at), "
        "extract(epoch from started_at) FROM tasks "
        f"WHERE enqueued_at >= to_timestamp({t0}) AND finished_at <= to_timestamp({t_end}) "
        "ORDER BY finished_at"
    )
    out = compose(
        "exec",
        "-T",
        "-e",
        f"Q={query}",
        "db",
        "sh",
        "-c",
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F, -c "$Q"',
    )
    rows = []
    for line in out.stdout.splitlines():
        finished, enqueued, started = (float(x) for x in line.split(","))
        rows.append(
            {
                "finished_t": round(finished - t0, 3),
                "in_system_s": round(finished - enqueued, 3),
                "inference_s": round(finished - started, 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare(args.queue)
    subprocess.run(["docker", "rm", "-f", SOURCE_CONTAINER], capture_output=True, check=False)

    stats_rows: list[dict] = []
    stop_stats = threading.Event()
    series: list[dict] = []
    scaler: subprocess.Popen | None = None
    scaled_at = None

    compose(
        "--profile",
        "load",
        "run",
        "-d",
        "--name",
        SOURCE_CONTAINER,
        "source",
        "--count",
        str(args.count),
        "--batch-id",
        args.batch_id,
        "--max-steps",
        str(args.max_steps),
    )
    t0 = time.time()
    stats_thread = threading.Thread(target=docker_stats_loop, args=(stop_stats, t0, stats_rows))
    stats_thread.start()
    try:
        while time.time() - t0 < args.duration:
            tick = time.time()
            series.append(sample(t0))
            if scaler is None and tick - t0 >= args.scale_at:
                scaled_at = round(tick - t0, 1)
                scaler = subprocess.Popen(
                    [
                        "docker",
                        "compose",
                        "up",
                        "-d",
                        "--scale",
                        "worker=2",
                        "--no-recreate",
                        "worker",
                    ],
                    cwd=ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            time.sleep(max(0.0, 1.0 - (time.time() - tick)))
        series.append(sample(t0))
    finally:
        t_end = time.time()
        subprocess.run(
            ["docker", "stop", "-t", "2", SOURCE_CONTAINER], capture_output=True, check=False
        )
        stop_stats.set()
        stats_thread.join()

    progress = source_progress(t0)
    tasks = task_times(t0, t_end)
    if not progress:
        raise SystemExit("no source_progress lines in the source log: the source did not run")
    subprocess.run(["docker", "rm", "-f", SOURCE_CONTAINER], capture_output=True, check=False)
    prefix = OUT_DIR / args.label
    write_csv(Path(f"{prefix}_queue.csv"), series)
    write_csv(Path(f"{prefix}_containers.csv"), stats_rows)
    write_csv(Path(f"{prefix}_source.csv"), progress)
    write_csv(Path(f"{prefix}_tasks.csv"), tasks)
    Path(f"{prefix}_meta.json").write_text(
        json.dumps(
            {
                "label": args.label,
                "queue": args.queue,
                "max_length": args.max_length,
                "batch_id": args.batch_id,
                "max_steps": args.max_steps,
                "t0": datetime.fromtimestamp(t0, UTC).isoformat(),
                "scale_at_s": scaled_at,
                "duration_s": args.duration,
                "tasks_done": len(tasks),
                "source_published_total": progress[-1]["published_total"] if progress else 0,
                "source_rejected_total": progress[-1]["rejected_total"] if progress else 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    last = series[-1]
    print(
        f"{args.label}: done={len(tasks)} ready={last['messages_ready']} "
        f"unacked={last['messages_unacknowledged']} "
        f"published_total={progress[-1]['published_total'] if progress else 0}"
    )


def cleanup(args: argparse.Namespace) -> None:
    """Delete the scenario's queues; app and worker come back with the values from .env."""
    compose("stop", "worker")
    delete_queues(args.queue)
    restart_pipeline()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run one scenario")
    runner.add_argument("--label", required=True)
    runner.add_argument("--queue", required=True)
    runner.add_argument("--max-length", type=int, required=True)
    runner.add_argument("--batch-id", required=True)
    # Two inference threads share one CPU-limited container and one GIL: with the default 200
    # steps an episode takes ~0.9 s instead of 0.14 s and the 0.5 s of INFERENCE_SECONDS stops
    # being the bottleneck. 10 steps keep the compute well under 0.5 s.
    runner.add_argument("--max-steps", type=int, default=10)
    runner.add_argument("--count", type=int, default=1_000_000)
    runner.add_argument("--duration", type=float, default=180)
    runner.add_argument("--scale-at", type=float, default=90)
    clean = commands.add_parser(
        "cleanup", help="delete a scenario's queues and restore the defaults"
    )
    clean.add_argument("--queue", required=True)
    args = parser.parse_args()

    if args.command == "run":
        os.environ["TASKS_QUEUE"] = args.queue
        os.environ["QUEUE_MAX_LENGTH"] = str(args.max_length)
        ENV["TASKS_QUEUE"] = args.queue
        run(args)
    else:
        cleanup(args)


if __name__ == "__main__":
    main()
