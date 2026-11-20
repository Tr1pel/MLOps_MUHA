"""Task submission and status endpoints.

The client of ``POST /api/v1/tasks`` is outside our control and cannot be slowed down, so
the API never waits for room in the queue: it makes one publish attempt and, if the broker
refuses it, answers 503 immediately and leaves nothing behind in the database.
"""

import math
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

from agent_platform.core.config import Settings
from agent_platform.core.logging import logger, task_id_context
from agent_platform.db.tasks import get_task, insert_pending
from agent_platform.messaging import (
    BrokerUnavailableError,
    QueueFullError,
    TaskMessage,
    publish_task,
)
from agent_platform.sim.env import DEFAULT_MAX_STEPS

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


def retry_after_seconds(settings: Settings) -> int:
    """Seconds after which a rejected client may reasonably retry.

    A full queue holds ``queue_max_length`` tasks and the worker finishes
    ``worker_concurrency`` tasks per ``inference_seconds``, so draining the whole queue takes
    ``queue_max_length * inference_seconds / worker_concurrency``. Retrying earlier is
    pointless: the freed slots are taken by other clients and a task accepted now would wait
    in the queue for that long anyway. Without a length limit there is no backlog to wait
    for, so one task duration is used. The result is at least 1 (a whole number of seconds).
    """
    backlog = settings.queue_max_length if settings.queue_max_length > 0 else 1
    return max(1, math.ceil(backlog * settings.inference_seconds / settings.worker_concurrency))


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: Annotated[int, Field(ge=0, le=2**32 - 1, description="Episode (world) seed")]
    max_steps: Annotated[int, Field(ge=1, le=1000)] = DEFAULT_MAX_STEPS


class TaskAccepted(BaseModel):
    task_id: UUID
    status: Literal["PENDING"] = "PENDING"


class TaskRejected(BaseModel):
    error: Literal["queue_full", "broker_unavailable"]


class TaskNotFound(BaseModel):
    error: Literal["not_found"] = "not_found"


class TaskStatus(BaseModel):
    task_id: UUID
    status: Literal["PENDING", "DONE"]
    seed: int
    max_steps: int
    model_version: str | None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict | None = Field(
        description="Episode result (seed, success, steps, collisions, trajectory, food, "
        "obstacles) once the task is DONE, otherwise null."
    )


def _reject(settings: Settings, error: str) -> JSONResponse:
    return JSONResponse(
        {"error": error},
        status_code=503,
        headers={"Retry-After": str(retry_after_seconds(settings))},
    )


@router.post(
    "",
    status_code=202,
    response_model=TaskAccepted,
    responses={
        503: {
            "model": TaskRejected,
            "description": "Queue is full or the broker is unavailable; nothing was stored. "
            "Retry after the number of seconds in the Retry-After header.",
            "headers": {"Retry-After": {"description": "Seconds", "schema": {"type": "integer"}}},
        }
    },
)
async def create_task(payload: TaskRequest, request: Request):
    """Queue one episode for the worker. Answers 202 only once the broker confirmed the message."""
    settings = request.app.state.settings
    message = TaskMessage(
        task_id=uuid4(),
        seed=payload.seed,
        max_steps=payload.max_steps,
        enqueued_at=datetime.now(UTC),
    )
    task_id_context.set(str(message.task_id))
    span = trace.get_current_span()
    span.set_attribute("task.id", str(message.task_id))
    # Publish first: if the broker refuses, there is no row to roll back.
    try:
        await publish_task(request.app.state.broker, settings, message)
    except (QueueFullError, BrokerUnavailableError) as exc:
        reason = "queue_full" if isinstance(exc, QueueFullError) else "broker_unavailable"
        retry_after = retry_after_seconds(settings)
        # The 503 is a deliberate answer, but for the trace it is a failed request.
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, reason))
        logger.warning("task_rejected", extra={"reason": reason, "retry_after": retry_after})
        return _reject(settings, reason)
    try:
        await insert_pending(request.app.state.engine, message)
    except Exception as exc:
        # The task is already in the queue and the worker upserts its result row, so it will
        # not be lost; failing the request here would make the client resubmit a duplicate.
        logger.warning("pending_insert_failed", extra={"error_type": type(exc).__name__})
    return JSONResponse(
        TaskAccepted(task_id=message.task_id).model_dump(mode="json"),
        status_code=202,
        headers={"Location": f"/api/v1/tasks/{message.task_id}"},
    )


@router.get(
    "/{task_id}",
    response_model=TaskStatus,
    responses={404: {"model": TaskNotFound, "description": "Unknown task"}},
)
async def read_task(task_id: UUID, request: Request):
    record = await get_task(request.app.state.engine, task_id)
    if record is None:
        return JSONResponse(TaskNotFound().model_dump(), status_code=404)
    return TaskStatus(**vars(record))
