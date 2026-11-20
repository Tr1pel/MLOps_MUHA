import asyncio
import threading
import time
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from conftest import FakePolicy, make_loaded
from faststream.rabbit import TestRabbitBroker
from faststream.rabbit.message import RabbitMessage
from pydantic import ValidationError

import agent_platform.worker as worker_module
from agent_platform.core.config import Settings
from agent_platform.messaging import TaskMessage
from agent_platform.ml.inference import ModelLoadError
from agent_platform.worker import TaskWorker, infer

QUEUE = "tasks"


class BrokenPolicy:
    def __init__(self):
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        raise RuntimeError("model exploded")


def body(**overrides):
    fields = {
        "task_id": str(uuid4()),
        "seed": 3,
        "max_steps": 5,
        "enqueued_at": datetime.now(UTC).isoformat(),
    }
    return {**fields, **overrides}


def make_message():
    return TaskMessage.model_validate(body())


@pytest.fixture
def acks(monkeypatch):
    """Record which acknowledgement the worker's message ended with."""
    events = []
    for name in ("ack", "nack", "reject"):
        original = getattr(RabbitMessage, name)

        def spy(original=original, name=name):
            async def wrapper(self, *args, **kwargs):
                events.append(name)
                return await original(self, *args, **kwargs)

            return wrapper

        monkeypatch.setattr(RabbitMessage, name, spy())
    return events


@pytest.fixture
def saved():
    return []


@pytest.fixture
def delivery():
    """The delivery number the fake counter reports; tests set ``delivery.n``."""

    class Delivery:
        n = 1
        registered = []

    return Delivery()


@pytest.fixture
def make_worker(saved, delivery):
    async def register(engine, task_id):
        delivery.registered.append(task_id)
        return delivery.n

    async def save(engine, message, result, model_version, started_at, finished_at):
        saved.append((message, result, model_version, started_at, finished_at))
        return True

    def factory(model=None, **settings):
        settings = Settings(_env_file=None, **{"inference_seconds": 0.0, **settings})
        worker = TaskWorker(
            settings,
            model_loader=lambda _: make_loaded(model),
            engine_factory=lambda _: None,
            save=save,
            register=register,
        )
        worker.loaded = make_loaded(model, version="7")
        worker.engine = object()
        return worker

    return factory


async def test_successful_task_is_saved_then_acked(make_worker, saved, acks, delivery):
    worker = make_worker(FakePolicy())
    payload = body()
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(payload, queue=QUEUE)

    [(message, result, version, started_at, finished_at)] = saved
    assert str(message.task_id) == payload["task_id"]
    assert version == "7"
    assert result["seed"] == 3
    assert result["steps"] <= 5
    assert started_at <= finished_at
    assert acks == ["ack"]
    assert delivery.registered == [message.task_id]


async def test_failure_before_the_limit_is_nacked_with_requeue(make_worker, saved, delivery):
    model = BrokenPolicy()
    worker = make_worker(model, task_delivery_limit=3)
    delivery.n = 2
    requeues = []
    original = RabbitMessage.nack

    async def spy(self, *args, **kwargs):
        requeues.append(kwargs.get("requeue"))
        return await original(self, *args, **kwargs)

    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RabbitMessage, "nack", spy)
            await broker.publish(body(), queue=QUEUE)

    assert model.calls == 1
    assert saved == []
    assert requeues == [True]


async def test_failure_on_the_last_delivery_is_rejected_to_dlq(make_worker, saved, acks, delivery):
    model = BrokenPolicy()
    worker = make_worker(model, task_delivery_limit=3)
    delivery.n = 3
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(RuntimeError, match="model exploded"):
            await broker.publish(body(), queue=QUEUE)

    assert model.calls == 1
    assert saved == []
    assert acks == ["reject"]


async def test_delivery_over_the_limit_is_rejected_without_running(
    make_worker, saved, acks, delivery
):
    model = BrokenPolicy()
    worker = make_worker(model, task_delivery_limit=3)
    delivery.n = 4
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(body(), queue=QUEUE)

    assert model.calls == 0
    assert saved == []
    assert acks == ["reject"]


async def test_save_error_is_nacked_and_the_redelivery_is_acked_after_the_save(
    make_worker, saved, acks, delivery
):
    worker = make_worker(FakePolicy(), task_delivery_limit=3)
    real_save = worker._save
    attempts = []

    async def flaky(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("db down")
        return await real_save(*args)

    worker._save = flaky
    payload = body()
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(payload, queue=QUEUE)
        assert (saved, acks) == ([], ["nack"])
        delivery.n = 2  # what the broker's redelivery would be counted as
        await broker.publish(payload, queue=QUEUE)

    assert len(saved) == 1
    assert acks == ["nack", "ack"]


async def test_failing_delivery_counter_rejects_the_message(make_worker, saved, acks):
    worker = make_worker(FakePolicy())

    async def broken(engine, task_id):
        raise ConnectionError("db down")

    worker._register = broken
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(ConnectionError):
            await broker.publish(body(), queue=QUEUE)

    assert saved == []
    assert acks == ["reject"]


@pytest.mark.parametrize("payload", [{"seed": 1}, body(seed=-5), body(extra="x")])
async def test_invalid_message_is_rejected_without_saving(make_worker, saved, acks, payload):
    worker = make_worker(FakePolicy())
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(ValidationError):
            await broker.publish(payload, queue=QUEUE)

    assert saved == []
    assert acks == ["reject"]


def test_infer_pads_to_target_duration_and_returns_plain_data():
    started = time.perf_counter()
    result = infer(FakePolicy(), make_message(), 0.3)
    assert time.perf_counter() - started >= 0.3
    assert isinstance(result["trajectory"], list)


def test_infer_does_not_sleep_when_the_episode_is_already_slow_enough():
    started = time.perf_counter()
    infer(FakePolicy(), make_message(), 0.0)
    assert time.perf_counter() - started < 0.2


async def test_inference_runs_in_pool_threads_without_blocking_the_event_loop(
    make_worker, monkeypatch
):
    worker = make_worker(FakePolicy(), worker_concurrency=2, inference_seconds=0.4)
    threads = set()
    original = worker_module.infer

    def spy(*args):
        threads.add(threading.current_thread().name)
        return original(*args)

    monkeypatch.setattr(worker_module, "infer", spy)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        started = time.perf_counter()
        await asyncio.gather(*(worker.handle(make_message()) for _ in range(2)))
        elapsed = time.perf_counter() - started
    finally:
        task.cancel()

    assert elapsed < 0.7  # two 0.4 s tasks overlapped, so concurrency really is 2
    assert ticks >= 10  # the event loop kept running during inference
    assert len(threads) == 2
    assert all(name.startswith("inference") for name in threads)


def test_one_setting_drives_prefetch_and_pool_size(monkeypatch):
    captured = {}
    real = worker_module.make_broker

    def spy(settings, **kwargs):
        captured.update(kwargs)
        return real(settings, **kwargs)

    monkeypatch.setattr(worker_module, "make_broker", spy)
    worker = TaskWorker(Settings(_env_file=None, worker_concurrency=3, inference_seconds=1.0))
    assert captured["prefetch_count"] == 3
    assert worker._executor._max_workers == 3
    assert captured["graceful_timeout"] > 2.0  # longer than two inferences


async def test_startup_prepares_schema_topology_and_loads_model(monkeypatch):
    steps = []

    async def fake_schema(engine):
        steps.append("schema")

    async def fake_topology(broker, settings):
        steps.append("topology")

    async def fake_connect():
        steps.append("connect")

    monkeypatch.setattr(worker_module, "ensure_schema", fake_schema)
    monkeypatch.setattr(worker_module, "declare_topology", fake_topology)
    worker = TaskWorker(
        Settings(_env_file=None),
        model_loader=lambda _: (steps.append("model"), make_loaded(version="9"))[1],
        engine_factory=lambda _: None,
    )
    monkeypatch.setattr(worker.broker, "connect", fake_connect)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker.app), base_url="http://worker"
    ) as client:
        before = await client.get("/healthz")
        await worker.startup()
        after = await client.get("/healthz")

    assert steps == ["schema", "connect", "topology", "model"]
    assert before.status_code == 503
    assert before.json() == {"status": "starting", "model_version": None}
    assert after.status_code == 200
    assert after.json() == {"status": "ok", "model_version": "9"}


async def test_startup_fails_when_model_cannot_be_loaded(monkeypatch):
    async def noop(*args):
        return None

    monkeypatch.setattr(worker_module, "ensure_schema", noop)
    monkeypatch.setattr(worker_module, "declare_topology", noop)

    def failing_loader(_):
        raise ModelLoadError("alias_not_found")

    worker = TaskWorker(
        Settings(_env_file=None), model_loader=failing_loader, engine_factory=lambda _: None
    )
    monkeypatch.setattr(worker.broker, "connect", noop)
    with pytest.raises(ModelLoadError):
        await worker.startup()
    assert worker.loaded is None


async def test_cleanup_disposes_engine_and_pool(make_worker):
    disposed = []

    class Engine:
        async def dispose(self):
            disposed.append(True)

    worker = make_worker(FakePolicy())
    worker.engine = Engine()
    await worker.cleanup()
    assert disposed == [True]
    with pytest.raises(RuntimeError):
        worker._executor.submit(lambda: None)


def test_main_serves_the_health_endpoint_on_the_configured_port(monkeypatch):
    captured = {}
    monkeypatch.setattr(worker_module.uvicorn, "run", lambda app, **kw: captured.update(kw))
    monkeypatch.setenv("WORKER_HEALTH_PORT", "8123")
    worker_module.main()
    assert captured["port"] == 8123
    assert captured["host"] == "0.0.0.0"
