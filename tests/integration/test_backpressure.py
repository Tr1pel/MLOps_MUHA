"""Backpressure pipeline against a real RabbitMQ and PostgreSQL started by Testcontainers."""

import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import aio_pika
import pytest
from conftest import FakePolicy, make_loaded
from opentelemetry import trace
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from agent_platform import source
from agent_platform.core.config import Settings
from agent_platform.db.connection import create_engine
from agent_platform.db.tasks import (
    ensure_schema,
    get_task,
    insert_pending,
    register_delivery,
    save_result,
)
from agent_platform.messaging import (
    QueueFullError,
    TaskMessage,
    dead_letter_queue_name,
    declare_topology,
    make_broker,
    publish_task,
)
from agent_platform.worker import TaskWorker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1"),
]

RABBIT_IMAGE = "rabbitmq:4.1.8-management"
POSTGRES_IMAGE = "postgres:17.6-bookworm"


@pytest.fixture(scope="module")
def rabbit():
    container = (
        DockerContainer(RABBIT_IMAGE)
        .with_exposed_ports(5672)
        .waiting_for(LogMessageWaitStrategy("Server startup complete").with_startup_timeout(120))
    )
    with container:
        yield container.get_container_host_ip(), int(container.get_exposed_port(5672))


@pytest.fixture(scope="module")
def postgres():
    with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
        yield container


@pytest.fixture
def settings(rabbit):
    host, port = rabbit
    # A fresh queue name per test keeps tests independent inside one broker container.
    return Settings(
        _env_file=None,
        rabbitmq_host=host,
        rabbitmq_port=port,
        tasks_queue=f"tasks-{uuid4().hex[:8]}",
        queue_max_length=5,
        inference_seconds=0.1,
        worker_concurrency=1,
        publish_timeout_seconds=5,
    )


@pytest.fixture
async def monitor(settings):
    """Separate connection used only to read queue depths (passive declare = ready count; a robust channel would return a stale cached count)."""
    connection = await aio_pika.connect(settings.rabbitmq_url)
    channel = await connection.channel()

    async def ready(name: str) -> int:
        queue = await channel.declare_queue(name, passive=True)
        return queue.declaration_result.message_count

    yield ready
    await connection.close()


def new_message(seed: int = 1) -> TaskMessage:
    return TaskMessage(task_id=uuid4(), seed=seed, max_steps=5, enqueued_at=datetime.now(UTC))


async def first_delivery(engine, task_id):
    return 1


def build_worker(settings, model, save, engine=None):
    """Without an engine the delivery counter is faked; with one it is the real Postgres upsert."""
    worker = TaskWorker(
        settings,
        model_loader=lambda _: make_loaded(model),
        engine_factory=lambda _: None,
        save=save,
        register=register_delivery if engine is not None else first_delivery,
    )
    worker.loaded = make_loaded(model)
    worker.engine = engine if engine is not None else object()
    return worker


async def wait_until(predicate, deadline=20.0):
    async with asyncio.timeout(deadline):
        while not await predicate():  # noqa: ASYNC110 (polls the broker, nothing to await)
            await asyncio.sleep(0.05)


async def test_full_queue_never_exceeds_limit_and_every_task_is_delivered_once(settings, monitor):
    delivered = []

    async def save(engine, message, result, model_version, started_at, finished_at):
        delivered.append(message.task_id)
        return True

    producer = make_broker(settings)
    await producer.start()
    await declare_topology(producer, settings)
    worker = build_worker(settings, FakePolicy(), save)  # one slot, 0.1 s per task: slow
    await worker.broker.start()

    depths = []
    stop_sampling = asyncio.Event()

    async def sample():
        while not stop_sampling.is_set():
            depths.append(await monitor(settings.tasks_queue))
            await asyncio.sleep(0.005)

    sampler = asyncio.create_task(sample())
    rejections = 0

    async def publish(message):
        nonlocal rejections
        try:
            await publish_task(producer, settings, message)
        except QueueFullError:
            rejections += 1
            raise

    tasks = list(source.generate_tasks(30, 0, "it-batch", 5))
    try:
        published, rejected = await source.run_source(iter(tasks), publish)
        await wait_until(lambda: _done(delivered, 30))
    finally:
        stop_sampling.set()
        await sampler
        await worker.broker.stop()
        await producer.stop()

    assert published == 30
    assert rejected == rejections > 0  # the source really was pushed back
    assert max(depths) <= settings.queue_max_length
    assert max(depths) >= 1
    assert len(delivered) == 30  # no duplicates...
    assert set(delivered) == {task.task_id for task in tasks}  # ...and no losses


async def _done(delivered, expected):
    return len(delivered) >= expected


async def _deliveries(engine, task_id):
    async with engine.connect() as connection:
        return (
            await connection.execute(
                text("SELECT deliveries FROM task_deliveries WHERE task_id = :id"), {"id": task_id}
            )
        ).scalar_one()


async def test_always_failing_consumer_ends_in_dlq_after_the_delivery_limit(
    settings, monitor, engine
):
    class AlwaysFails:
        calls = 0

        def predict(self, X):
            AlwaysFails.calls += 1
            raise RuntimeError("boom")

    async def save(*args):
        raise AssertionError("must not be saved")

    settings = settings.model_copy(update={"task_delivery_limit": 3, "inference_seconds": 0})
    producer = make_broker(settings)
    await producer.start()
    await declare_topology(producer, settings)
    worker = build_worker(settings, AlwaysFails(), save, engine)
    await worker.broker.start()
    dlq = dead_letter_queue_name(settings)
    message = new_message()
    try:
        await publish_task(producer, settings, message)
        await wait_until(lambda: _count_is(monitor, dlq, 1))
        await asyncio.sleep(0.3)  # an endless requeue loop would show up as extra calls here
        assert await monitor(settings.tasks_queue) == 0
        assert await monitor(dlq) == 1
    finally:
        await worker.broker.stop()
        await producer.stop()
    assert AlwaysFails.calls == 3
    assert await _deliveries(engine, message.task_id) == 3


async def _crash_on_delivery(settings, engine):
    """Take one message and drop the connection without acking: what a SIGKILLed worker does."""
    connection = await aio_pika.connect(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        queue = await channel.declare_queue(settings.tasks_queue, passive=True)
        async with asyncio.timeout(10):
            while (delivered := await queue.get(fail=False)) is None:  # noqa: ASYNC110 (polls the broker)
                await asyncio.sleep(0.05)
        task = TaskMessage.model_validate_json(delivered.body)
        await register_delivery(engine, task.task_id)  # the worker counts before it runs
    finally:
        await connection.close()


async def test_message_that_crashes_the_worker_goes_to_dlq_without_running(
    settings, monitor, engine
):
    class MustNotRun:
        calls = 0

        def predict(self, X):
            MustNotRun.calls += 1
            return 0

    async def save(*args):
        raise AssertionError("must not be saved")

    settings = settings.model_copy(update={"task_delivery_limit": 3})
    producer = make_broker(settings)
    await producer.start()
    await declare_topology(producer, settings)
    dlq = dead_letter_queue_name(settings)
    message = new_message()
    try:
        await publish_task(producer, settings, message)
        for _ in range(settings.task_delivery_limit):
            await _crash_on_delivery(settings, engine)
        await wait_until(lambda: _count_is(monitor, settings.tasks_queue, 1))  # back in the queue

        worker = build_worker(settings, MustNotRun(), save, engine)
        await worker.broker.start()
        try:
            await wait_until(lambda: _count_is(monitor, dlq, 1))
        finally:
            await worker.broker.stop()
    finally:
        await producer.stop()
    assert await monitor(settings.tasks_queue) == 0
    assert MustNotRun.calls == 0
    assert await _deliveries(engine, message.task_id) == settings.task_delivery_limit + 1


async def test_poison_message_is_retried_in_one_trace_then_goes_to_dlq_without_saving(
    settings, monitor, engine, spans
):
    saved = []

    async def save(*args):
        saved.append(args)
        return True

    settings = settings.model_copy(update={"task_delivery_limit": 3})
    producer = make_broker(settings)
    await producer.start()
    await declare_topology(producer, settings)
    worker = build_worker(settings, FakePolicy(), save, engine)
    await worker.broker.start()
    dlq = dead_letter_queue_name(settings)
    task_id = uuid4()
    try:
        with trace.get_tracer("test").start_as_current_span("send") as send:
            await producer.publish(
                {"task_id": str(task_id), "seed": -1},
                queue=settings.tasks_queue,
                persist=True,
                correlation_id=str(task_id),
            )
        await wait_until(lambda: _count_is(monitor, dlq, 1))
        assert await monitor(settings.tasks_queue) == 0
    finally:
        await worker.broker.stop()
        await producer.stop()
    assert saved == []
    assert await _deliveries(engine, task_id) == 3

    finished = spans.get_finished_spans()
    process = [s for s in finished if s.name.endswith(" process")]
    trace_id = send.get_span_context().trace_id
    assert sorted(s.attributes["messaging.delivery_attempt"] for s in process) == [1, 2, 3]
    assert {s.context.trace_id for s in process} == {trace_id}  # redeliveries stay in the trace
    assert any(e.name == "dead_lettered" for s in process for e in s.events)
    database = [s for s in finished if s.attributes.get("db.system") == "postgresql"]
    assert database
    assert {s.context.trace_id for s in database} >= {trace_id}  # DB spans join the same trace


async def _count_is(monitor, queue, expected):
    return await monitor(queue) == expected


async def test_declaring_topology_twice_is_idempotent(settings):
    broker = make_broker(settings)
    await broker.start()
    try:
        await declare_topology(broker, settings)
        await declare_topology(broker, settings)
    finally:
        await broker.stop()


@pytest.fixture
async def engine(postgres):
    settings = Settings(
        _env_file=None,
        postgres_host=postgres.get_container_host_ip(),
        postgres_port=int(postgres.get_exposed_port(5432)),
        postgres_db=postgres.dbname,
        postgres_user=postgres.username,
        postgres_password=postgres.password,
    )
    engine = create_engine(settings)
    await ensure_schema(engine)
    yield engine
    await engine.dispose()


async def test_save_result_is_idempotent_on_real_postgres(engine):
    message = new_message(seed=42)
    started, finished = datetime.now(UTC), datetime.now(UTC)

    assert await save_result(engine, message, {"steps": 3}, "5", started, finished) is True
    assert await save_result(engine, message, {"steps": 99}, "6", started, finished) is False

    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text("SELECT count(*) FROM tasks WHERE task_id = :id"), {"id": message.task_id}
            )
        ).scalar_one()
    assert rows == 1
    record = await get_task(engine, message.task_id)
    assert record.status == "DONE"
    assert record.result == {"steps": 3}  # the second call changed nothing
    assert record.model_version == "5"
    assert record.seed == 42


async def test_insert_pending_never_rolls_back_a_finished_task(engine):
    message = new_message()
    await save_result(engine, message, {"ok": True}, "1", datetime.now(UTC), datetime.now(UTC))
    await insert_pending(engine, message)
    assert (await get_task(engine, message.task_id)).status == "DONE"


async def test_pending_row_is_completed_once_by_the_worker(engine):
    message = new_message()
    await insert_pending(engine, message)
    await insert_pending(engine, message)
    pending = await get_task(engine, message.task_id)
    assert (pending.status, pending.result, pending.finished_at) == ("PENDING", None, None)

    assert await save_result(engine, message, {"ok": 1}, "2", datetime.now(UTC), datetime.now(UTC))
    done = await get_task(engine, message.task_id)
    assert (done.status, done.result, done.model_version) == ("DONE", {"ok": 1}, "2")
    assert done.finished_at is not None


async def test_unknown_task_is_none_and_concurrent_ensure_schema_is_safe(engine):
    assert await get_task(engine, uuid4()) is None
    await asyncio.gather(*(ensure_schema(engine) for _ in range(5)))
