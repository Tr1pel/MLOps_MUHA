"""Exercise a running API with the stdlib; works locally and in CI."""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import urlopen


def fetch(base_url: str, path: str) -> tuple[int, dict]:
    try:
        response = urlopen(f"{base_url}{path}", timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--database-down", action="store_true")
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
    print(json.dumps({"liveness": live, "build": version, "health": health}, indent=2))


if __name__ == "__main__":
    main()
