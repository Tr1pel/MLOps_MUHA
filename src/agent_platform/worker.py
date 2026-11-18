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
too, so a message that kills the worker also ends up in the DLQ, without being run again. A
message that cannot be parsed as a ``TaskMessage`` is rejected straight away: retrying cannot
fix it.
"""

import asyncio
import dataclasses
import json
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import uvicorn
from faststream.asgi import AsgiFastStream, AsgiResponse, get
from faststream.exceptions import NackMessage, RejectMessage
from faststream.middlewares import AckPolicy
from sqlalchemy.ext.asyncio import AsyncEngine

from agent_platform.core.config import Settings
from agent_platform.core.logging import configure_logging, logger
from agent_platform.db.connection import create_engine
from agent_platform.db.tasks import ensure_schema, register_delivery, save_result
from agent_platform.messaging import TaskMessage, declare_topology, make_broker, tasks_queue
from agent_platform.ml.inference import (
    LoadedModel,
    ModelLoader,
    _episode,
    _run_in_daemon_thread,
    load_registered_model,
)

SaveResult = Callable[..., Awaitable[bool]]
RegisterDelivery = Callable[..., Awaitable[int]]
EngineFactory = Callable[[Settings], AsyncEngine]


def infer(model: Any, message: TaskMessage, inference_seconds: float) -> dict:
    """Blocking: run the episode, then pad to ``inference_seconds`` in total.

    Runs in the thread pool, so the sleep never touches the event loop.
    """
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
            after_shutdown=[self.cleanup],
        )

    async def startup(self) -> None:
        """Runs before the broker starts consuming; any failure aborts the process."""
        self.engine = self._engine_factory(self.settings)
        await ensure_schema(self.engine)
        await self.broker.connect()
        await declare_topology(self.broker, self.settings)
        # Blocking MLflow calls must stay off the event loop; a hung one must not block exit.
        self.loaded = await _run_in_daemon_thread(
            lambda: self._model_loader(self.settings), self.settings.model_load_timeout_seconds
        )
        logger.info(
            "worker_started",
            extra={
                "model_name": self.loaded.info.name,
                "model_alias": self.loaded.info.alias,
                "model_version": self.loaded.info.version,
            },
        )

    async def cleanup(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.engine is not None:
            await self.engine.dispose()

    async def _healthz(self) -> AsgiResponse:
        loaded = self.loaded
        body = {"status": "ok" if loaded else "starting"}
        body["model_version"] = loaded.info.version if loaded else None
        return AsgiResponse(
            json.dumps(body).encode(),
            200 if loaded else 503,
            headers={"content-type": "application/json"},
        )

    async def handle(self, message: TaskMessage) -> None:
        limit = self.settings.task_delivery_limit
        assert self.engine is not None, "handler ran before startup finished"
        # If counting itself fails (database down) the exception propagates: reject, no requeue.
        delivery = await self._register(self.engine, message.task_id)
        if delivery > limit:
            logger.error(
                "task_delivery_limit_exceeded task_id=%s delivery=%d/%d",
                message.task_id,
                delivery,
                limit,
            )
            raise RejectMessage()
        try:
            await self._process(message)
        except Exception as exc:
            logger.error(
                "task_failed task_id=%s delivery=%d/%d",
                message.task_id,
                delivery,
                limit,
                extra={"error_type": type(exc).__name__},
            )
            if delivery < limit:
                raise NackMessage(requeue=True) from exc
            raise

    async def _process(self, message: TaskMessage) -> None:
        loaded, engine = self.loaded, self.engine
        assert loaded is not None and engine is not None, "handler ran before startup finished"
        started_at = datetime.now(UTC)
        started = perf_counter()
        result = await asyncio.get_running_loop().run_in_executor(
            self._executor, infer, loaded.model, message, self.settings.inference_seconds
        )
        finished_at = datetime.now(UTC)
        written = await self._save(
            engine, message, result, loaded.info.version, started_at, finished_at
        )
        logger.info(
            "task_done task_id=%s written=%s",
            message.task_id,
            written,
            extra={
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "model_version": loaded.info.version,
            },
        )


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    worker = TaskWorker(settings)
    # uvicorn.run (not AsgiFastStream.run) exits non-zero when startup fails, e.g. no model.
    uvicorn.run(
        worker.app,
        host="0.0.0.0",
        port=settings.worker_health_port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
