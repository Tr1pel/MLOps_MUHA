import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from aio_pika.exceptions import AMQPException
from aiormq.exceptions import ChannelInvalidStateError, DeliveryError
from pydantic import ValidationError

from agent_platform import messaging
from agent_platform.core.config import Settings
from agent_platform.messaging import (
    BrokerUnavailableError,
    QueueFullError,
    TaskMessage,
    declare_topology,
    make_broker,
    publish_task,
    tasks_queue,
)


def make_message(**overrides):
    fields = {"task_id": uuid4(), "seed": 7, "max_steps": 10, "enqueued_at": datetime.now(UTC)}
    return TaskMessage(**{**fields, **overrides})


class RecordingBroker:
    def __init__(self, error=None, delay=0.0):
        self.error, self.delay, self.published, self.declared = error, delay, [], []

    async def publish(self, body, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        self.published.append((body, kwargs))

    async def declare_exchange(self, exchange):
        self.declared.append(("exchange", exchange))
        return RecordingBound(self)

    async def declare_queue(self, queue):
        self.declared.append(("queue", queue))
        return RecordingBound(self)


class RecordingBound:
    def __init__(self, broker):
        self.broker = broker

    async def bind(self, exchange, routing_key):
        self.broker.declared.append(("bind", routing_key))


def test_rabbitmq_url_escapes_credentials():
    settings = Settings(
        _env_file=None,
        rabbitmq_host="mq",
        rabbitmq_port=5673,
        rabbitmq_user="u@ser",
        rabbitmq_password="p@ss/w:rd",
    )
    assert settings.rabbitmq_url == "amqp://u%40ser:p%40ss%2Fw%3Ard@mq:5673"


@pytest.mark.parametrize(
    "fields",
    [{"seed": -1}, {"seed": 2**32}, {"max_steps": 0}, {"max_steps": 1001}, {"extra": 1}],
)
def test_task_message_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        make_message(**fields)


def test_task_message_defaults_max_steps():
    message = TaskMessage(task_id=uuid4(), seed=1, enqueued_at=datetime.now(UTC))
    assert message.max_steps == 200


def test_main_queue_is_bounded_and_dead_lettered():
    queue = tasks_queue(Settings(_env_file=None, tasks_queue="jobs", queue_max_length=5))
    assert queue.name == "jobs"
    assert queue.durable
    assert queue.arguments == {
        "x-queue-type": "classic",
        "x-dead-letter-exchange": "jobs.dlx",
        "x-dead-letter-routing-key": "jobs.dlq",
        "x-max-length": 5,
        "x-overflow": "reject-publish",
    }


def test_zero_max_length_means_unbounded_queue():
    queue = tasks_queue(Settings(_env_file=None, queue_max_length=0))
    assert "x-max-length" not in queue.arguments
    assert "x-overflow" not in queue.arguments


def test_make_broker_enables_confirms_and_prefetch(monkeypatch):
    captured = {}

    def fake_broker(url, **kwargs):
        captured.update(url=url, **kwargs)

    monkeypatch.setattr(messaging, "RabbitBroker", fake_broker)
    settings = Settings(_env_file=None)
    make_broker(settings, prefetch_count=3, graceful_timeout=9.0)
    assert captured["url"] == settings.rabbitmq_url
    assert captured["graceful_timeout"] == 9.0
    assert captured["default_channel"].publisher_confirms is True
    assert captured["default_channel"].prefetch_count == 3
    assert any(type(m).__name__ == "RabbitTelemetryMiddleware" for m in captured["middlewares"])


async def test_trace_context_travels_with_the_message_and_the_consumer_continues_it(spans):
    from faststream.rabbit import TestRabbitBroker
    from faststream.rabbit.annotations import RabbitMessage
    from opentelemetry import trace

    settings = Settings(_env_file=None, tasks_queue="jobs-trace")
    broker = make_broker(settings)
    seen = []

    @broker.subscriber(settings.tasks_queue)
    async def consume(message: RabbitMessage):
        seen.append((message.headers, message.raw_message.correlation_id))

    message = make_message()
    async with TestRabbitBroker(broker):
        with trace.get_tracer("t").start_as_current_span("root") as root:
            await publish_task(broker, settings, message)

    [(headers, correlation_id)] = seen
    assert correlation_id == str(message.task_id)
    trace_id = format(root.get_span_context().trace_id, "032x")
    assert headers["traceparent"].split("-")[1] == trace_id
    names = {span.name: span for span in spans.get_finished_spans()}
    process = next(span for name, span in names.items() if name.endswith("process"))
    assert format(process.context.trace_id, "032x") == trace_id


async def test_declare_topology_declares_dlx_dlq_binding_and_queue():
    broker = RecordingBroker()
    await declare_topology(broker, Settings(_env_file=None, tasks_queue="jobs"))
    kinds = [kind for kind, _ in broker.declared]
    assert kinds == ["exchange", "queue", "bind", "queue"]
    assert broker.declared[0][1].name == "jobs.dlx"
    assert broker.declared[1][1].name == "jobs.dlq"
    assert broker.declared[2][1] == "jobs.dlq"
    assert broker.declared[3][1].name == "jobs"


async def test_publish_task_sends_one_persistent_confirmed_message():
    broker = RecordingBroker()
    settings = Settings(_env_file=None, tasks_queue="jobs", publish_timeout_seconds=1.5)
    message = make_message()
    await publish_task(broker, settings, message)
    [(body, kwargs)] = broker.published
    assert body["task_id"] == str(message.task_id)
    assert TaskMessage.model_validate(body) == message
    assert kwargs == {
        "queue": "jobs",
        "persist": True,
        "timeout": 1.5,
        "correlation_id": str(message.task_id),
    }


async def test_nack_from_broker_is_queue_full():
    broker = RecordingBroker(error=DeliveryError(None, None))
    with pytest.raises(QueueFullError):
        await publish_task(broker, Settings(_env_file=None), make_message())


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError(), ChannelInvalidStateError("closed"), AMQPException("down")],
)
async def test_connection_errors_are_broker_unavailable(error):
    with pytest.raises(BrokerUnavailableError):
        await publish_task(RecordingBroker(error=error), Settings(_env_file=None), make_message())


async def test_slow_confirm_times_out_as_broker_unavailable():
    settings = Settings(_env_file=None, publish_timeout_seconds=0.05)
    with pytest.raises(BrokerUnavailableError):
        await publish_task(RecordingBroker(delay=5), settings, make_message())
