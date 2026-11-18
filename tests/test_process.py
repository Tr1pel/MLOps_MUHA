import asyncio
import threading

import pytest
from conftest import FakePolicy, make_loaded


def loader_for(loaded, calls=None):
    def load(settings):
        if calls is not None:
            calls.append(settings.model_alias)
        return loaded

    return load


class Slow(FakePolicy):
    """Blocks inside predict until the test opens the gate."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def predict(self, X):
        self.entered.set()
        self.gate.wait(5)
        return super().predict(X)


async def wait_until(event: threading.Event):
    assert await asyncio.to_thread(event.wait, 2)


async def test_process_returns_episode_and_model_block(model_settings, started_app):
    async with started_app(model_settings, loader_for(make_loaded())) as (_, client):
        response = await client.post("/process", json={"seed": 7, "max_steps": 20})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "seed",
        "success",
        "steps",
        "collisions",
        "trajectory",
        "food",
        "obstacles",
        "model",
    }
    assert body["seed"] == 7
    assert 1 <= body["steps"] <= 20
    assert len(body["trajectory"]) == body["steps"] + 1
    assert len(body["food"]) == 2
    assert all(set(o) == {"x", "y", "radius"} for o in body["obstacles"])
    assert body["model"] == {
        "name": "fly-policy",
        "alias": "champion",
        "version": "3",
        "run_id": "run-abc",
    }


async def test_default_max_steps_and_determinism(model_settings, started_app):
    async with started_app(model_settings, loader_for(make_loaded())) as (_, client):
        first = await client.post("/process", json={"seed": 1})
        second = await client.post("/process", json={"seed": 1})
    assert first.status_code == 200
    assert first.json()["steps"] <= 200
    assert first.json() == second.json()


async def test_model_is_loaded_once_across_requests(model_settings, started_app):
    calls = []
    policy = FakePolicy()
    async with started_app(model_settings, loader_for(make_loaded(policy), calls)) as (_, client):
        for seed in range(4):
            assert (await client.post("/process", json={"seed": seed, "max_steps": 5})).is_success
        await client.get("/api/v1/model")
    assert calls == ["champion"]
    assert policy.calls >= 4


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"seed": "x"},
        {"seed": -1},
        {"seed": 1, "max_steps": 0},
        {"seed": 1, "max_steps": 1001},
        {"seed": 1, "max_steps": "many"},
        {"seed": 1, "extra": True},
    ],
)
async def test_invalid_input_is_422(model_settings, started_app, payload):
    async with started_app(model_settings, loader_for(make_loaded())) as (_, client):
        response = await client.post("/process", json=payload)
    assert response.status_code == 422


async def test_503_when_mlflow_not_configured(settings, started_app):
    async with started_app(settings, loader_for(make_loaded())) as (_, client):
        response = await client.post("/process", json={"seed": 1})
    assert response.status_code == 503
    assert response.json() == {"error": "model_unavailable", "reason": "not_configured"}


async def test_503_when_load_failed_and_app_still_starts(model_settings, started_app, capsys):
    def broken(settings):
        raise ConnectionError("http://user:hunter2@mlflow.test refused")

    async with started_app(model_settings, broken) as (_, client):
        assert (await client.get("/healthz")).status_code == 200
        response = await client.post("/process", json={"seed": 1})
    assert response.status_code == 503
    assert response.json() == {"error": "model_unavailable", "reason": "load_failed"}
    out = capsys.readouterr().out
    assert "ConnectionError" in out
    assert "hunter2" not in out
    assert "hunter2" not in response.text


async def test_503_when_load_times_out(model_settings, started_app):
    release = threading.Event()
    settings = model_settings.model_copy(update={"model_load_timeout_seconds": 0.05})

    def hung(settings):
        release.wait(5)
        return make_loaded()

    try:
        async with started_app(settings, hung) as (_, client):
            response = await client.post("/process", json={"seed": 1})
            status = await client.get("/api/v1/model")
    finally:
        release.set()
    assert response.status_code == 503
    assert response.json()["reason"] == "timeout"
    assert status.json()["reason"] == "timeout"


async def test_healthz_answers_while_episode_runs(model_settings, started_app):
    slow = Slow()
    async with started_app(model_settings, loader_for(make_loaded(slow))) as (_, client):
        pending = asyncio.create_task(client.post("/process", json={"seed": 1, "max_steps": 3}))
        try:
            await wait_until(slow.entered)
            live = await asyncio.wait_for(client.get("/healthz"), timeout=1)
            assert live.status_code == 200
            assert not pending.done()
        finally:
            slow.gate.set()
        assert (await pending).status_code == 200


async def test_busy_when_slots_exhausted_and_slot_survives_cancellation(
    model_settings, started_app
):
    slow = Slow()
    settings = model_settings.model_copy(update={"inference_concurrency": 1})
    async with started_app(settings, loader_for(make_loaded(slow))) as (_, client):
        first = asyncio.create_task(client.post("/process", json={"seed": 1, "max_steps": 2}))
        try:
            await wait_until(slow.entered)
            busy = await client.post("/process", json={"seed": 2})
            assert busy.status_code == 503
            assert busy.json() == {"error": "busy", "reason": None}
            assert busy.headers["retry-after"] == "1"
            # Client gives up; the worker thread is still running, so the slot stays taken.
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert (await client.post("/process", json={"seed": 3})).status_code == 503
        finally:
            slow.gate.set()
        response = None
        for _ in range(200):
            response = await client.post("/process", json={"seed": 4, "max_steps": 2})
            if response.status_code == 200:
                break
            await asyncio.sleep(0.01)
        assert response.status_code == 200


@pytest.mark.parametrize("failure", [RuntimeError("secret-xyz"), None])
async def test_inference_failure_is_clean_500(model_settings, started_app, capsys, failure):
    class Broken(FakePolicy):
        def predict(self, X):
            if failure:
                raise failure
            return [9]  # not a valid action id

    async with started_app(model_settings, loader_for(make_loaded(Broken()))) as (_, client):
        response = await client.post("/process", json={"seed": 1})
    assert response.status_code == 500
    assert response.json() == {"error": "inference_failed", "reason": None}
    out = capsys.readouterr().out
    assert "inference_failed" in out
    assert ("RuntimeError" if failure else "ValueError") in out
    assert "secret-xyz" not in out + response.text


async def test_model_endpoint_reports_loaded_version(model_settings, started_app):
    async with started_app(model_settings, loader_for(make_loaded(version="5"))) as (_, client):
        body = (await client.get("/api/v1/model")).json()
    assert body == {
        "loaded": True,
        "name": "fly-policy",
        "alias": "champion",
        "version": "5",
        "run_id": "run-abc",
        "source": "mlflow-artifacts:/1/abc/artifacts/model",
        "loaded_at": "2026-01-01T00:00:00Z",
        "load_duration_ms": 12.5,
        "reason": None,
    }


async def test_model_endpoint_reports_not_loaded(client):
    body = (await client.get("/api/v1/model")).json()
    assert body["loaded"] is False
    assert body["reason"] == "not_configured"
    assert body["version"] is None
    assert body["name"] == "fly-policy" and body["alias"] == "champion"


async def test_health_contract_is_unchanged_by_model(client):
    response = await client.get("/api/v1/health")
    assert set(response.json()["components"]) == {"postgresql"}


async def test_restart_picks_up_new_alias_target(model_settings, started_app):
    registry = {"champion": "1"}

    def load(settings):
        return make_loaded(version=registry[settings.model_alias])

    async with started_app(model_settings, load) as (_, old_client):
        registry["champion"] = "2"  # alias moved while the first instance runs
        before = await old_client.post("/process", json={"seed": 1, "max_steps": 2})
        async with started_app(model_settings, load) as (_, new_client):
            after = await new_client.post("/process", json={"seed": 1, "max_steps": 2})
        still_old = await old_client.post("/process", json={"seed": 1, "max_steps": 2})
    assert before.json()["model"]["version"] == "1"
    assert after.json()["model"]["version"] == "2"
    assert still_old.json()["model"]["version"] == "1"
