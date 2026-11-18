"""FastStream worker: consume tasks from RabbitMQ, run the registered model, store results.

One setting, ``worker_concurrency``, drives three numbers: the channel prefetch count (how many
messages the broker lets this worker hold unacknowledged), the number of messages handled at
once (FastStream/aio-pika start one handler task per delivered message), and the size of the
inference thread pool. So the worker never holds more messages than it can run in parallel.

Acknowledgement: the message is acked after the result is stored (``REJECT_ON_ERROR``: a normal
return acks). A classic queue does not count deliveries, so every delivery is counted in Postgres
before the task runs. A failing task is nacked with requeue (a real redelivery by the broker)
until the ``task_delivery_limit``-th delivery; that one is rejected without requeue, which the
queue's dead-letter arguments route to ``<queue>.dlq``. Deliveries after a crashed process count
too, so a message that kills the worker also ends up in the DLQ, without being run again.

The handler takes the raw message and validates ``TaskMessage`` itself, so that an invalid
("poison") message is counted and retried exactly like a valid one: every attempt is a separate
consumer span in the one trace of the publish, and after the limit it goes to the DLQ. The
counter key is the message's ``correlation_id`` (the task id, set by the publisher) when it is a
UUID, otherwise a UUID derived from the SHA-256 of the body.

Observability: the consumer span comes from FastStream's telemetry middleware; the handler adds
attributes to it, an ``inference`` span (created inside the pool thread, which runs in a copy of
the handler's context) and a ``db.save_result`` span. ``task_id`` is put into a ``ContextVar``
for the duration of the handler, so every log record, from this code or from a library, has it.
"""

import asyncio
import contextvars
import dataclasses
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_OID, UUID, uuid5

import uvicorn
from faststream.asgi import AsgiFastStream, AsgiResponse, get
from faststream.exceptions import NackMessage, RejectMessage
from faststream.middlewares import AckPolicy
from faststream.rabbit.annotations import RabbitMessage
from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from agent_platform.core.config import Settings
from agent_platform.core.logging import configure_logging, logger, task_id_context
from agent_platform.core.telemetry import setup_telemetry, shutdown_telemetry
from agent_platform.db.connection import create_engine
from agent_platform.db.tasks import ensure_schema, register_delivery, save_result
from agent_platform.messaging import TaskMessage, declare_topology, make_broker, tasks_queue
from agent_platform.ml.inference import (
    LoadedModel,
    ModelInfo,
    ModelLoader,
    _episode,
    _run_in_daemon_thread,
    load_registered_model,
)

SERVICE_NAME = "worker"
tracer = trace.get_tracer("agent_platform.worker")
meter = metrics.get_meter("agent_platform.worker")
processed_counter = meter.create_counter("worker.tasks.processed", unit="{task}")
inference_histogram = meter.create_histogram("worker.inference.duration", unit="s")
task_histogram = meter.create_histogram("worker.task.duration", unit="s")
in_progress_counter = meter.create_up_down_counter("worker.tasks.in_progress", unit="{task}")
redelivery_counter = meter.create_counter("worker.redeliveries", unit="{delivery}")

SaveResult = Callable[..., Awaitable[bool]]
RegisterDelivery = Callable[..., Awaitable[int]]
EngineFactory = Callable[[Settings], AsyncEngine]


def delivery_key(message: RabbitMessage) -> UUID:
    """Key of the delivery counter: the publisher's correlation id, else a hash of the body.

    ``message.correlation_id`` is no use here: FastStream invents a new one on every receive when
    the wire message has none, which would reset the count on each redelivery.
    """
    correlation_id = message.raw_message.correlation_id
    if correlation_id:
        try:
            return UUID(correlation_id)
        except ValueError:
            pass
    return uuid5(NAMESPACE_OID, hashlib.sha256(message.body).hexdigest())


def infer(
    model: Any, message: TaskMessage, inference_seconds: float, model_info: ModelInfo | None = None
) -> dict:
    """Blocking: run the episode, then pad to ``inference_seconds`` in total.

    Runs in the thread pool, so the sleep never touches the event loop. The span is created here,
    in the pool thread; it is a child of the handler's span because the caller runs this function
    in a copy of the handler's context.
    """
    attributes: dict[str, str | int] = {
        "task.seed": message.seed,
        "task.max_steps": message.max_steps,
    }
    if model_info is not None:
        attributes |= {
            "model.name": model_info.name,
            "model.alias": model_info.alias,
            "model.version": model_info.version,
        }
    with tracer.start_as_current_span("inference", attributes=attributes):
        started = perf_counter()
        episode = _episode(model, message.seed, message.max_steps)
        remaining = inference_seconds - (perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)
        return dataclasses.asdict(episode)


class TaskWorker:
    def __init__(
        self,
        settings: Settings,
        *,
        model_loader: ModelLoader = load_registered_model,
        engine_factory: EngineFactory = create_engine,
        save: SaveResult = save_result,
        register: RegisterDelivery = register_delivery,
    ) -> None:
        self.settings = settings
        self._model_loader = model_loader
        self._engine_factory = engine_factory
        self._save = save
        self._register = register
        self.loaded: LoadedModel | None = None
        self.engine: AsyncEngine | None = None
        self.in_flight = 0
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_concurrency, thread_name_prefix="inference"
        )
        self.broker = make_broker(
            settings,
            prefetch_count=settings.worker_concurrency,
            # Long enough for tasks already taken to finish and be acked after SIGTERM.
            graceful_timeout=settings.inference_seconds * 2 + 5,
        )
        self.broker.subscriber(tasks_queue(settings), ack_policy=AckPolicy.REJECT_ON_ERROR)(
            self.handle
        )
        self.app = AsgiFastStream(
            self.broker,
            asgi_routes=[("/healthz", get(self._healthz))],
            logger=None,
            on_startup=[self.startup],
            on_shutdown=[self.shutdown],
            after_shutdown=[self.cleanup],
        )

    async def startup(self) -> None:
        """Runs before the broker starts consuming; any failure aborts the process."""
        try:
            await self._startup()
        except Exception:
            # uvicorn prints the traceback too, but as one unstructured event text.
            logger.error("worker_startup_failed", exc_info=True)
            raise

    async def _startup(self) -> None:
        self.engine = self._engine_factory(self.settings)
        await ensure_schema(self.engine)
        await self.broker.connect()
        await declare_topology(self.broker, self.settings)
        # Blocking MLflow calls must stay off the event loop; a hung one must not block exit.
        self.loaded = await _run_in_daemon_thread(
            lambda: self._model_loader(self.settings), self.settings.model_load_timeout_seconds
        )
        info = self.loaded.info
        model_fields = {
            "model_name": info.name,
            "model_alias": info.alias,
            "model_version": info.version,
        }
        logger.info("model_loaded", extra={**model_fields, "duration_ms": info.load_duration_ms})
        logger.info("worker_started", extra=model_fields)

    async def shutdown(self) -> None:
        """Runs when SIGTERM arrives, before the broker stops waiting for tasks in progress."""
        logger.info("shutdown_started", extra={"in_flight": self.in_flight})

    async def cleanup(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.engine is not None:
            await self.engine.dispose()
        logger.info("shutdown_finished", extra={"in_flight": self.in_flight})
        await asyncio.to_thread(shutdown_telemetry)

    async def _healthz(self) -> AsgiResponse:
        loaded = self.loaded
        body = {"status": "ok" if loaded else "starting"}
        body["model_version"] = loaded.info.version if loaded else None
        return AsgiResponse(
            json.dumps(body).encode(),
            200 if loaded else 503,
            headers={"content-type": "application/json"},
        )

    async def handle(self, message: RabbitMessage) -> None:
        key = delivery_key(message)
        token = task_id_context.set(str(key))
        span = trace.get_current_span()
        span.set_attribute("task.id", str(key))
        self.in_flight += 1
        try:
            await self._handle(key, message.body, span)
        finally:
            self.in_flight -= 1
            task_id_context.reset(token)

    async def _handle(self, key: UUID, body: bytes, span: Span) -> None:
        limit = self.settings.task_delivery_limit
        assert self.engine is not None, "handler ran before startup finished"
        # If counting itself fails (database down) the exception propagates: reject, no requeue.
        delivery = await self._register(self.engine, key)
        span.set_attribute("messaging.delivery_attempt", delivery)
        if delivery > 1:
            redelivery_counter.add(1)
            span.add_event("redelivery", {"delivery": delivery})
        if delivery > limit:
            self._dead_letter(span, delivery, "delivery_limit_exceeded")
            raise RejectMessage()
        try:
            message = TaskMessage.model_validate_json(body)
        except ValidationError as exc:
            # Neither the log nor the span may carry the body: pydantic quotes the input.
            problems = [
                {"loc": ".".join(map(str, error["loc"])), "type": error["type"]}
                for error in exc.errors(include_input=False, include_url=False)
            ]
            logger.error(
                "task_invalid",
                extra={"delivery": delivery, "delivery_limit": limit, "problems": problems},
            )
            span.add_event("invalid_message", {"delivery": delivery})
            span.set_status(Status(StatusCode.ERROR, "invalid_message"))
            if delivery < limit:
                processed_counter.add(1, {"result": "error"})
                raise NackMessage(requeue=True) from None
            self._dead_letter(span, delivery, "invalid_message")
            raise RejectMessage() from None
        try:
            await self._process(message)
        except Exception as exc:
            logger.error(
                "task_failed",
                extra={
                    "delivery": delivery,
                    "delivery_limit": limit,
                    "error_type": type(exc).__name__,
                },
                exc_info=True,
            )
            if delivery < limit:
                processed_counter.add(1, {"result": "error"})
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise NackMessage(requeue=True) from exc
            self._dead_letter(span, delivery, "processing_failed")
            raise

    def _dead_letter(self, span: Span, delivery: int, reason: str) -> None:
        """The message is about to be rejected without requeue, i.e. sent to the DLQ."""
        processed_counter.add(1, {"result": "dlq"})
        span.add_event("dead_lettered", {"reason": reason, "delivery": delivery})
        span.set_status(Status(StatusCode.ERROR, reason))
        logger.error(
            "task_dead_lettered",
            extra={
                "delivery": delivery,
                "delivery_limit": self.settings.task_delivery_limit,
                "reason": reason,
            },
        )

    async def _process(self, message: TaskMessage) -> None:
        loaded, engine = self.loaded, self.engine
        assert loaded is not None and engine is not None, "handler ran before startup finished"
        started_at = datetime.now(UTC)
        started = perf_counter()
        queue_wait = (started_at - message.enqueued_at).total_seconds()
        trace.get_current_span().set_attribute("task.queue_wait_s", round(queue_wait, 3))
        in_progress_counter.add(1)
        try:
            # The pool thread does not inherit the context: run in a copy of ours so that the
            # inference span has the handler's span as its parent and logs keep trace/task ids.
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                contextvars.copy_context().run,
                infer,
                loaded.model,
                message,
                self.settings.inference_seconds,
                loaded.info,
            )
            finished_at = datetime.now(UTC)
            inference_histogram.record(perf_counter() - started)
            with tracer.start_as_current_span("db.save_result") as save_span:
                written = await self._save(
                    engine, message, result, loaded.info.version, started_at, finished_at
                )
                save_span.set_attribute("db.rows_written", int(written))
        finally:
            in_progress_counter.add(-1)
        task_histogram.record((datetime.now(UTC) - message.enqueued_at).total_seconds())
        processed_counter.add(1, {"result": "ok"})
        logger.info(
            "task_done",
            extra={
                "written": written,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "model_version": loaded.info.version,
            },
        )


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level, SERVICE_NAME)
    setup_telemetry(SERVICE_NAME, settings)
    worker = TaskWorker(settings)
    # uvicorn.run (not AsgiFastStream.run) exits non-zero when startup fails, e.g. no model.
    uvicorn.run(
        worker.app,
        host="0.0.0.0",
        port=settings.worker_health_port,
        # No uvicorn log config: its loggers propagate to the root JSON handler.
        log_config=None,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
