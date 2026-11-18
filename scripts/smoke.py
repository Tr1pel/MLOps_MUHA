"""Exercise a running API with the stdlib; works locally and in CI."""

import argparse
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TASK_TIMEOUT_SECONDS = 60


def fetch(base_url: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=30)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--database-down", action="store_true")
    parser.add_argument(
        "--tasks",
        action="store_true",
        help="also submit a task and wait for DONE (needs RabbitMQ, a worker and a trained model)",
    )
    parser.add_argument("--commit")
    parser.add_argument("--version")
    args = parser.parse_args()

    status, live = fetch(args.base_url, "/healthz")
    assert status == 200 and live == {"status": "ok"}, (status, live)
    status, version = fetch(args.base_url, "/api/v1/version")
    assert status == 200, (status, version)
    if args.commit:
        assert version["commit"] == args.commit, version
    if args.version:
        assert version["version"] == args.version, version
    status, health = fetch(args.base_url, "/api/v1/health")
    component = health["components"]["postgresql"]
    assert component["response_time_ms"] >= 0, component
    if args.database_down:
        assert status == 503 and health["status"] == "error", (status, health)
        assert component["status"] == "error" and component["version"] is None, component
    else:
        assert status == 200 and health["status"] == "ok", (status, health)
        assert component["status"] == "ok" and component["version"], component
    report = {"liveness": live, "build": version, "health": health}
    if args.tasks:
        status, accepted = fetch(args.base_url, "/api/v1/tasks", {"seed": 1, "max_steps": 50})
        assert status == 202 and accepted["status"] == "PENDING", (status, accepted)
        path = f"/api/v1/tasks/{accepted['task_id']}"
        deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
        while True:
            status, task = fetch(args.base_url, path)
            assert status == 200, (status, task)
            if task["status"] == "DONE":
                break
            assert time.monotonic() < deadline, f"task not DONE in {TASK_TIMEOUT_SECONDS}s: {task}"
            time.sleep(0.5)
        result = task["result"]
        assert 1 <= result["steps"] <= 50 and task["model_version"], task
        assert len(result["trajectory"]) == result["steps"] + 1, result
        assert isinstance(result["success"], bool) and result["collisions"] >= 0, result
        report["task"] = {key: value for key, value in task.items() if key != "result"}
        print(f"task {task['task_id']} done with model version {task['model_version']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
