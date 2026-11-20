"""Batch source: publish N tasks, one at a time, as fast as the broker accepts them.

Backpressure comes entirely from the broker: when the bounded queue is full the publish is
rejected, the source sleeps with exponential backoff and retries the *same* task. There is no
rate limit and no buffer: only the current task is held in memory.
"""

import argparse
import asyncio
import random
import signal
import sys
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from time import monotonic
from uuid import UUID, uuid5

from faststream.rabbit import RabbitBroker
from opentelemetry import metrics, trace

from agent_platform.core.config import Settings
from agent_platform.core.logging import configure_logging, logger, task_id_context
from agent_platform.core.telemetry import setup_telemetry, shutdown_telemetry
from agent_platform.messaging import (
    BrokerUnavailableError,
    QueueFullError,
    TaskMessage,
    declare_topology,
    make_broker,
    publish_task,
)
from agent_platform.sim.env import DEFAULT_MAX_STEPS

SERVICE_NAME = "source"
tracer = trace.get_tracer("agent_platform.source")
meter = metrics.get_meter("agent_platform.source")
published_counter = meter.create_counter("source.tasks.published", unit="{task}")
rejected_counter = meter.create_counter("source.publish.rejected", unit="{task}")
backoff_counter = meter.create_counter("source.backoff.duration", unit="s")

TASK_NAMESPACE = UUID("6f1b3c1e-8d0e-4b8e-9a53-2c6f0a1d7e42")
BACKOFF_BASE_SECONDS = 0.05
BACKOFF_FACTOR = 2.0
BACKOFF_CAP_SECONDS = 1.0
REPORT_INTERVAL_SECONDS = 1.0

Publish = Callable[[TaskMessage], Awaitable[None]]


def task_id_for(batch_id: str, seed: int) -> UUID:
    """Deterministic id: re-running a batch yields the same ids, so the database deduplicates."""
    return uuid5(TASK_NAMESPACE, f"{batch_id}:{seed}")


def generate_tasks(
    count: int, start_seed: int, batch_id: str, max_steps: int
) -> Iterator[TaskMessage]:
    """Lazily yield one record at a time; nothing is materialised."""
    for seed in range(start_seed, start_seed + count):
        yield TaskMessage(
            task_id=task_id_for(batch_id, seed),
            seed=seed,
            max_steps=max_steps,
            enqueued_at=datetime.now(UTC),
        )


def backoff_delay(failures: int) -> float:
    """Exponential delay (base 0.05 s, x2, cap 1 s) with up to 20% random jitter on top."""
    delay = min(BACKOFF_BASE_SECONDS * BACKOFF_FACTOR ** (failures - 1), BACKOFF_CAP_SECONDS)
    return delay * (1 + random.uniform(0, 0.2))


async def run_source(
    tasks: Iterator[TaskMessage],
    publish: Publish,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    report_interval: float = REPORT_INTERVAL_SECONDS,
) -> tuple[int, int]:
    """Publish every task once the broker confirms it. Returns (published, rejected)."""
    published = rejected = 0
    last_report = monotonic()
    published_at_last_report = 0

    def report(now: float) -> None:
        nonlocal last_report, published_at_last_report
        rate = (published - published_at_last_report) / max(now - last_report, 1e-9)
        logger.info(
            "source_progress",
            extra={
                "published_total": published,
                "rejected_total": rejected,
                "rate_per_s": round(rate, 1),
            },
        )
        last_report, published_at_last_report = now, published

    for task in tasks:
        failures = 0
        # One trace per task (never one per batch): the publish span below is its child.
        with tracer.start_as_current_span(
            "source.publish_task",
            attributes={"task.id": str(task.task_id), "task.seed": task.seed},
        ) as span:
            token = task_id_context.set(str(task.task_id))
            try:
                while True:
                    try:
                        # The task counts as enqueued when the broker accepts it, not when it
                        # was generated.
                        await publish(task.model_copy(update={"enqueued_at": datetime.now(UTC)}))
                        break
                    except (QueueFullError, BrokerUnavailableError) as exc:
                        rejected += 1
                        failures += 1
                        delay = backoff_delay(failures)
                        reason = "queue_full" if isinstance(exc, QueueFullError) else "unavailable"
                        span.add_event("broker_rejected", {"attempt": failures, "reason": reason})
                        span.add_event("backoff", {"attempt": failures, "delay_s": delay})
                        logger.warning(
                            "publish_rejected",
                            extra={
                                "attempt": failures,
                                "delay_s": round(delay, 3),
                                "reason": reason,
                            },
                        )
                        rejected_counter.add(1)
                        backoff_counter.add(delay)
                        await sleep(delay)
                    if monotonic() - last_report >= report_interval:
                        report(monotonic())
            finally:
                task_id_context.reset(token)
        published += 1
        published_counter.add(1)
        if monotonic() - last_report >= report_interval:
            report(monotonic())
    report(monotonic())
    return published, rejected


async def run(
    settings: Settings, count: int, start_seed: int, batch_id: str, max_steps: int
) -> int:
    broker: RabbitBroker = make_broker(settings)
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    assert main_task is not None
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, main_task.cancel)
        except NotImplementedError:  # Windows event loops have no add_signal_handler
            pass

    async def publish(message: TaskMessage) -> None:
        await publish_task(broker, settings, message)

    try:
        await broker.start()
        await declare_topology(broker, settings)
        logger.info(
            "source_started",
            extra={
                "count": count,
                "start_seed": start_seed,
                "batch_id": batch_id,
                "max_steps": max_steps,
            },
        )
        published, rejected = await run_source(
            generate_tasks(count, start_seed, batch_id, max_steps), publish
        )
        logger.info("source_done", extra={"published_total": published, "rejected_total": rejected})
        return 0
    except asyncio.CancelledError:
        logger.info("source_interrupted")
        return 130
    finally:
        await broker.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a batch of simulation tasks")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--batch-id", default="default")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)
    if args.count < 0 or args.start_seed < 0 or args.start_seed + args.count > 2**32:
        parser.error("--count and --start-seed must keep seeds within 0..2**32-1")
    if not 1 <= args.max_steps <= 1000:
        parser.error("--max-steps must be in 1..1000")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level, SERVICE_NAME)
    setup_telemetry(SERVICE_NAME, settings)
    try:
        return asyncio.run(
            run(settings, args.count, args.start_seed, args.batch_id, args.max_steps)
        )
    except KeyboardInterrupt:
        return 130
    finally:
        # A short-lived process: whatever is still buffered must leave before it exits.
        shutdown_telemetry()


if __name__ == "__main__":
    sys.exit(main())
