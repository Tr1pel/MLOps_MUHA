import asyncio
import importlib
from time import perf_counter
from unittest.mock import AsyncMock

import pytest

health_module = importlib.import_module("agent_platform.api.health")


async def test_liveness_does_not_check_database(client, monkeypatch):
    probe = AsyncMock(side_effect=RuntimeError("must not connect"))
    monkeypatch.setattr(health_module, "postgres_version", probe)
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]
    probe.assert_not_awaited()


async def test_health_reports_server_version_and_latency(client, monkeypatch):
    probe = AsyncMock(return_value="17.6 (Debian)")
    monkeypatch.setattr(health_module, "postgres_version", probe)
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    component = body["components"]["postgresql"]
    assert component["status"] == "ok"
    assert component["version"] == "17.6 (Debian)"
    assert component["response_time_ms"] >= 0
    assert component["error"] is None
    probe.assert_awaited_once()


@pytest.mark.parametrize("exception", [OSError, RuntimeError])
async def test_database_failure_is_sanitized(client, monkeypatch, log_output, exception):
    secret = "postgresql://agent:super-secret@db/private"
    monkeypatch.setattr(health_module, "postgres_version", AsyncMock(side_effect=exception(secret)))
    response = await client.get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    component = response.json()["components"]["postgresql"]
    assert component["status"] == "error"
    assert component["version"] is None
    assert component["error"] == "unavailable"
    assert component["response_time_ms"] >= 0
    assert secret not in response.text
    logs = log_output.getvalue()
    assert "dependency_check_failed" in logs
    assert exception.__name__ in logs
    assert secret not in logs


async def test_timeout_cancels_probe_and_returns_503(client, app, monkeypatch):
    cancelled = asyncio.Event()

    async def slow_probe(engine):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(health_module, "postgres_version", slow_probe)
    started = perf_counter()
    response = await client.get("/api/v1/health")
    elapsed = perf_counter() - started
    assert response.status_code == 503
    component = response.json()["components"]["postgresql"]
    assert component["error"] == "timeout"
    assert component["version"] is None
    assert component["response_time_ms"] >= app.state.settings.health_timeout_seconds * 900
    assert elapsed < 1
    assert cancelled.is_set()


async def test_liveness_responds_while_database_probe_is_waiting(client, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def waiting_probe(engine):
        entered.set()
        await release.wait()
        return "17.6"

    monkeypatch.setattr(health_module, "postgres_version", waiting_probe)
    pending = asyncio.create_task(client.get("/api/v1/health"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        live = await client.get("/healthz")
        assert live.status_code == 200
        assert not pending.done()
    finally:
        release.set()
        response = await pending
    assert response.status_code == 200
    assert response.headers["x-request-id"] != live.headers["x-request-id"]


async def test_external_cancellation_is_not_converted_to_dependency_error(client, monkeypatch):
    monkeypatch.setattr(
        health_module, "postgres_version", AsyncMock(side_effect=asyncio.CancelledError())
    )
    with pytest.raises(asyncio.CancelledError):
        await client.get("/api/v1/health")
