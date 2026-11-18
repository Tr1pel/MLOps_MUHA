import asyncio
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from conftest import SPAN_EXPORTER, FakePolicy, log_records, make_loaded, metric_total
from faststream.rabbit import TestRabbitBroker
from faststream.rabbit.message import RabbitMessage
from opentelemetry import trace
from opentelemetry.trace import StatusCode

import agent_platform.worker as worker_module
from agent_platform.core.config import Settings
from agent_platform.core.logging import task_id_context
from agent_platform.messaging import TaskMessage
from agent_platform.ml.inference import ModelLoadError
from agent_platform.worker import TaskWorker, delivery_key, infer

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


async def publish(broker, payload, **kwargs):
    """Publish like the real publishers: the task id is the correlation id."""
    kwargs.setdefault("correlation_id", payload.get("task_id"))
    await broker.publish(payload, queue=QUEUE, **kwargs)


def spans_named(suffix):
    return [s for s in SPAN_EXPORTER.get_finished_spans() if s.name.endswith(suffix)]


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
        await publish(broker, payload)

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


INVALID = [{"seed": 1}, body(seed=-5), body(extra="x")]


@pytest.mark.parametrize("payload", INVALID)
async def test_invalid_message_is_nacked_before_the_limit(
    make_worker, saved, acks, delivery, payload, log_output
):
    worker = make_worker(FakePolicy(), task_delivery_limit=3)
    delivery.n = 2
    async with TestRabbitBroker(worker.broker) as broker:
        await publish(broker, payload, correlation_id=str(uuid4()))

    assert saved == []
    assert acks == ["nack"]
    [record] = [r for r in log_records(log_output) if r["event"] == "task_invalid"]
    assert (record["delivery"], record["delivery_limit"]) == (2, 3)
    assert record["problems"]
    assert all(set(problem) == {"loc", "type"} for problem in record["problems"])  # no input


@pytest.mark.parametrize("payload", INVALID)
async def test_invalid_message_on_the_last_delivery_is_rejected_to_dlq(
    make_worker, saved, acks, delivery, payload, log_output
):
    worker = make_worker(FakePolicy(), task_delivery_limit=3)
    delivery.n = 3
    async with TestRabbitBroker(worker.broker) as broker:
        await publish(broker, payload)

    assert saved == []
    assert acks == ["reject"]
    [dead] = [r for r in log_records(log_output) if r["event"] == "task_dead_lettered"]
    assert (dead["reason"], dead["delivery"], dead["level"]) == ("invalid_message", 3, "ERROR")


async def test_poison_message_gets_every_attempt_in_one_trace_then_the_dlq(
    make_worker, acks, delivery
):
    worker = make_worker(FakePolicy(), task_delivery_limit=3)
    counts: dict[UUID, int] = {}

    async def register(engine, key):
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    worker._register = register
    poison = {"task_id": str(uuid4()), "seed": -1}
    async with TestRabbitBroker(worker.broker) as broker:
        with trace.get_tracer("t").start_as_current_span("send") as send:
            for _ in range(3):  # what the broker's requeue would do
                await publish(broker, poison)

    assert acks == ["nack", "nack", "reject"]
    assert list(counts) == [UUID(poison["task_id"])]
    process = spans_named("process")
    assert len(process) == 3
    assert {s.context.trace_id for s in process} == {send.get_span_context().trace_id}
    attempts = [s.attributes["messaging.delivery_attempt"] for s in process]
    assert sorted(attempts) == [1, 2, 3]
    last = next(s for s in process if s.attributes["messaging.delivery_attempt"] == 3)
    assert "dead_lettered" in [e.name for e in last.events]
    assert last.status.status_code is StatusCode.ERROR
    assert all(s.attributes["task.id"] == poison["task_id"] for s in process)


def test_delivery_key_is_the_correlation_id_or_a_hash_of_the_body():
    task_id = uuid4()
    with_uuid = SimpleNamespace(raw_message=SimpleNamespace(correlation_id=str(task_id)), body=b"x")
    assert delivery_key(with_uuid) == task_id

    def without(correlation_id, body):
        return delivery_key(
            SimpleNamespace(raw_message=SimpleNamespace(correlation_id=correlation_id), body=body)
        )

    # Stable for the same body, different for another one, and a UUID either way.
    assert without(None, b"garbage") == without("not-a-uuid", b"garbage")
    assert without(None, b"garbage") != without(None, b"other")
    assert isinstance(without("", b"garbage"), UUID)


def test_infer_pads_to_target_duration_and_returns_plain_data():
    started = time.perf_counter()
    result = infer(FakePolicy(), make_message(), 0.3)
    assert time.perf_counter() - started >= 0.3
    assert isinstance(result["trajectory"], list)


def test_infer_does_not_sleep_when_the_episode_is_already_slow_enough():
    started = time.perf_counter()
    infer(FakePolicy(), make_message(), 0.0)
    assert time.perf_counter() - started < 0.2


def test_infer_span_describes_the_model_and_the_task():
    message = make_message()
    infer(FakePolicy(), message, 0.0, make_loaded(version="9").info)
    [span] = spans_named("inference")
    assert dict(span.attributes) == {
        "task.seed": message.seed,
        "task.max_steps": message.max_steps,
        "model.name": "fly-policy",
        "model.alias": "champion",
        "model.version": "9",
    }


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
        await asyncio.gather(*(worker._process(make_message()) for _ in range(2)))
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


async def test_startup_fails_when_model_cannot_be_loaded(monkeypatch, log_output):
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
    [record] = [r for r in log_records(log_output) if r["event"] == "worker_startup_failed"]
    assert record["level"] == "ERROR"
    assert "ModelLoadError: alias_not_found" in record["exception"]


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


class LoggingPolicy(FakePolicy):
    """Logs from the pool thread, like a library would."""

    def predict(self, X):
        logging.getLogger("some.library").info("predicting")
        return super().predict(X)


async def test_spans_inference_and_save_are_children_of_the_processing_span(make_worker):
    worker = make_worker(FakePolicy())
    payload = body(enqueued_at=(datetime.now(UTC) - timedelta(seconds=2)).isoformat())
    async with TestRabbitBroker(worker.broker) as broker:
        with trace.get_tracer("t").start_as_current_span("send") as send:
            await publish(broker, payload)

    [process] = spans_named("process")
    [inference] = spans_named("inference")
    [save] = spans_named("db.save_result")
    assert inference.parent.span_id == process.context.span_id
    assert save.parent.span_id == process.context.span_id
    assert {process.context.trace_id, inference.context.trace_id, save.context.trace_id} == {
        send.get_span_context().trace_id
    }
    assert process.attributes["messaging.delivery_attempt"] == 1
    assert process.attributes["task.id"] == payload["task_id"]
    assert 1.9 < process.attributes["task.queue_wait_s"] < 5
    assert save.attributes["db.rows_written"] == 1


async def test_logs_from_the_pool_thread_carry_task_and_trace_ids(make_worker, log_output):
    worker = make_worker(LoggingPolicy())
    payload = body()
    async with TestRabbitBroker(worker.broker) as broker:
        await publish(broker, payload)

    [process] = spans_named("process")
    threaded = [r for r in log_records(log_output) if r["logger"] == "some.library"]
    assert threaded
    assert {r["task_id"] for r in threaded} == {payload["task_id"]}
    assert {r["trace_id"] for r in threaded} == {format(process.context.trace_id, "032x")}
    [done] = [r for r in log_records(log_output) if r["event"] == "task_done"]
    assert done["task_id"] == payload["task_id"]
    assert done["written"] is True
    assert "task_id=" not in done["event"]
    assert task_id_context.get() is None


async def test_failed_task_logs_a_traceback_with_the_delivery_and_marks_the_span(
    make_worker, delivery, log_output
):
    worker = make_worker(BrokenPolicy(), task_delivery_limit=3)
    delivery.n = 2
    async with TestRabbitBroker(worker.broker) as broker:
        await publish(broker, body())

    [failed] = [r for r in log_records(log_output) if r["event"] == "task_failed"]
    assert failed["level"] == "ERROR"
    assert (failed["delivery"], failed["delivery_limit"]) == (2, 3)
    assert "RuntimeError: model exploded" in failed["exception"]
    assert failed["task_id"] and failed["trace_id"]
    [process] = spans_named("process")
    assert process.status.status_code is StatusCode.ERROR
    assert "exception" in [e.name for e in process.events]
    assert "redelivery" in [e.name for e in process.events]
    assert process.attributes["messaging.delivery_attempt"] == 2


async def test_processing_error_on_the_last_delivery_is_dead_lettered(
    make_worker, delivery, log_output
):
    worker = make_worker(BrokenPolicy(), task_delivery_limit=3)
    delivery.n = 3
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(RuntimeError):
            await publish(broker, body())

    [dead] = [r for r in log_records(log_output) if r["event"] == "task_dead_lettered"]
    assert (dead["reason"], dead["delivery"], dead["delivery_limit"]) == ("processing_failed", 3, 3)
    [process] = spans_named("process")
    assert "dead_lettered" in [e.name for e in process.events]


async def test_metrics_count_results_redeliveries_and_durations(make_worker, delivery):
    def snapshot():
        return {
            "ok": metric_total("worker.tasks.processed", result="ok"),
            "error": metric_total("worker.tasks.processed", result="error"),
            "dlq": metric_total("worker.tasks.processed", result="dlq"),
            "redeliveries": metric_total("worker.redeliveries"),
            "inference": metric_total("worker.inference.duration"),
            "task": metric_total("worker.task.duration"),
        }

    before = snapshot()
    async with TestRabbitBroker(make_worker(FakePolicy()).broker) as broker:
        await publish(broker, body())
    delivery.n = 2
    failing = make_worker(BrokenPolicy(), task_delivery_limit=3)
    async with TestRabbitBroker(failing.broker) as broker:
        await publish(broker, body())
    delivery.n = 3
    async with TestRabbitBroker(failing.broker) as broker:
        with pytest.raises(RuntimeError):
            await publish(broker, body())
    after = snapshot()

    assert {key: after[key] - before[key] for key in before} == {
        "ok": 1,
        "error": 1,
        "dlq": 1,
        "redeliveries": 2,
        "inference": 1,
        "task": 1,
    }
    assert metric_total("worker.tasks.in_progress") == 0


def test_metric_attributes_have_no_task_or_trace_ids():
    from conftest import METRIC_READER

    data = METRIC_READER.get_metrics_data()
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name.startswith("worker."):
                    for point in metric.data.data_points:
                        assert set(point.attributes) <= {"result"}


async def test_shutdown_events_report_tasks_in_flight(make_worker, log_output):
    worker = make_worker(FakePolicy())
    worker.in_flight = 2
    worker.engine = None
    await worker.shutdown()
    await worker.cleanup()
    events = {r["event"]: r for r in log_records(log_output)}
    assert events["shutdown_started"]["in_flight"] == 2
    assert events["shutdown_finished"]["in_flight"] == 2


async def test_startup_logs_model_loaded_with_alias_version_and_duration(monkeypatch, log_output):
    async def noop(*args):
        return None

    monkeypatch.setattr(worker_module, "ensure_schema", noop)
    monkeypatch.setattr(worker_module, "declare_topology", noop)
    worker = TaskWorker(
        Settings(_env_file=None),
        model_loader=lambda _: make_loaded(version="4"),
        engine_factory=lambda _: None,
    )
    monkeypatch.setattr(worker.broker, "connect", noop)
    await worker.startup()
    [loaded] = [r for r in log_records(log_output) if r["event"] == "model_loaded"]
    assert (loaded["model_alias"], loaded["model_version"]) == ("champion", "4")
    assert loaded["duration_ms"] == 12.5
