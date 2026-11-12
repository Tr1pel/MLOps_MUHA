"""Exercise a running API with the stdlib; works locally and in CI."""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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
        "--process",
        action="store_true",
        help="also run POST /process and require a loaded model (needs MLflow and a trained model)",
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
    if args.process:
        status, model = fetch(args.base_url, "/api/v1/model")
        assert status == 200 and model["loaded"] and model["version"], (status, model)
        status, episode = fetch(args.base_url, "/process", {"seed": 1, "max_steps": 50})
        assert status == 200, (status, episode)
        assert 1 <= episode["steps"] <= 50, episode
        assert len(episode["trajectory"]) == episode["steps"] + 1, episode
        assert isinstance(episode["success"], bool) and episode["collisions"] >= 0, episode
        assert episode["model"]["version"] == model["version"], (episode["model"], model)
        report["model"] = model
        report["process"] = {key: value for key, value in episode.items() if key != "trajectory"}
        print(f"model in use: {model['name']}@{model['alias']} version {model['version']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
