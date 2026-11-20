"""Publish one deliberately invalid task, to watch a "poison" message travel the pipeline.

The worker counts every delivery of it, nacks it back into the queue until the delivery limit
and then dead-letters it: all attempts are consumer spans of the one trace started here.
Broker address and logging come from the environment, as for the other services.
"""

import asyncio
import sys
from uuid import uuid4

from opentelemetry import trace

from agent_platform.core.config import Settings
from agent_platform.core.logging import configure_logging, logger
from agent_platform.core.telemetry import setup_telemetry, shutdown_telemetry
from agent_platform.messaging import declare_topology, make_broker

SERVICE_NAME = "source"


async def send_poison(settings: Settings) -> None:
    task_id = str(uuid4())
    broker = make_broker(settings)
    await broker.start()
    try:
        await declare_topology(broker, settings)
        with trace.get_tracer("agent_platform.scripts").start_as_current_span(
            "source.publish_poison", attributes={"task.id": task_id}
        ):
            # seed=-1 fails TaskMessage validation: the worker can never process it.
            await broker.publish(
                {"task_id": task_id, "seed": -1},
                queue=settings.tasks_queue,
                persist=True,
                correlation_id=task_id,
                timeout=settings.publish_timeout_seconds,
            )
            # Inside the span, so the record carries the trace_id to look up.
            logger.info("poison_sent", extra={"task_id": task_id})
    finally:
        await broker.stop()


def main() -> int:
    settings = Settings()
    configure_logging(settings.log_level, SERVICE_NAME)
    setup_telemetry(SERVICE_NAME, settings)
    try:
        asyncio.run(send_poison(settings))
        return 0
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    sys.exit(main())
