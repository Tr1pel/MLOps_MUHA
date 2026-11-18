"""RabbitMQ topology and publishing shared by the API, the source and the worker.

Queue type: a *classic* durable queue, not a quorum one. Measured on RabbitMQ 4.1.8, a quorum
queue with ``x-overflow=reject-publish`` does not hold ``x-max-length``: with concurrent
publishers it accepted 20 messages for a limit of 5 (and 80 of 500 attempts at 100-way
concurrency), because quorum queues learn about the limit asynchronously. A classic queue held
the limit exactly in every run. Classic queues have no ``x-delivery-limit``, so the worker
counts attempts itself (see ``worker.py``) and rejects the message into the dead-letter queue.
"""

import asyncio
from datetime import datetime
from typing import Annotated
from uuid import UUID

from aio_pika.exceptions import AMQPException
from aiormq.exceptions import AMQPError, ChannelInvalidStateError, DeliveryError
from faststream.rabbit import ExchangeType, RabbitBroker, RabbitExchange, RabbitQueue
from faststream.rabbit.schemas import Channel
from pydantic import BaseModel, ConfigDict, Field

from agent_platform.core.config import Settings
from agent_platform.sim.env import DEFAULT_MAX_STEPS


class TaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    max_steps: Annotated[int, Field(ge=1, le=1000)] = DEFAULT_MAX_STEPS
    enqueued_at: datetime


class QueueFullError(Exception):
    """The broker nacked the publish (queue at x-max-length with reject-publish)."""


class BrokerUnavailableError(Exception):
    """The publish timed out or the connection to the broker is down."""


def dead_letter_exchange_name(settings: Settings) -> str:
    return f"{settings.tasks_queue}.dlx"


def dead_letter_queue_name(settings: Settings) -> str:
    return f"{settings.tasks_queue}.dlq"


def tasks_queue(settings: Settings) -> RabbitQueue:
    """The one description of the main queue; every declaring side must use it verbatim."""
    arguments: dict[str, str | int] = {
        "x-dead-letter-exchange": dead_letter_exchange_name(settings),
        "x-dead-letter-routing-key": dead_letter_queue_name(settings),
    }
    if settings.queue_max_length > 0:
        arguments["x-max-length"] = settings.queue_max_length
        arguments["x-overflow"] = "reject-publish"
    return RabbitQueue(settings.tasks_queue, durable=True, arguments=arguments)


def make_broker(
    settings: Settings,
    *,
    prefetch_count: int | None = None,
    graceful_timeout: float | None = None,
) -> RabbitBroker:
    """Broker whose channel uses publisher confirms (the aio-pika/FastStream default, made explicit)."""
    options = {} if graceful_timeout is None else {"graceful_timeout": graceful_timeout}
    return RabbitBroker(
        settings.rabbitmq_url,
        default_channel=Channel(prefetch_count=prefetch_count, publisher_confirms=True),
        # The project logs through agent_platform.core.logging, not FastStream's own logger.
        logger=None,
        **options,
    )


async def declare_topology(broker: RabbitBroker, settings: Settings) -> None:
    """Idempotently declare the main queue, the dead-letter exchange and the DLQ bound to it."""
    exchange = await broker.declare_exchange(
        RabbitExchange(dead_letter_exchange_name(settings), type=ExchangeType.DIRECT, durable=True)
    )
    dlq_name = dead_letter_queue_name(settings)
    dlq = await broker.declare_queue(RabbitQueue(dlq_name, durable=True))
    await dlq.bind(exchange, routing_key=dlq_name)
    await broker.declare_queue(tasks_queue(settings))


async def publish_task(broker: RabbitBroker, settings: Settings, message: TaskMessage) -> None:
    """One publish attempt of a persistent message, waiting for the broker's confirm.

    Retrying is the caller's decision: the source backs off, the API answers 503.
    """
    timeout = settings.publish_timeout_seconds
    try:
        async with asyncio.timeout(timeout):
            await broker.publish(
                message.model_dump(mode="json"),
                queue=settings.tasks_queue,
                persist=True,
                timeout=timeout,
            )
    except DeliveryError as exc:
        raise QueueFullError(settings.tasks_queue) from exc
    except (TimeoutError, AMQPError, AMQPException, ChannelInvalidStateError, OSError) as exc:
        raise BrokerUnavailableError(type(exc).__name__) from exc
