from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    Text,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.schema import CreateTable

from agent_platform.messaging import TaskMessage

PENDING = "PENDING"
DONE = "DONE"

tasks = Table(
    "tasks",
    MetaData(),
    Column("task_id", PG_UUID(as_uuid=True), primary_key=True),
    Column("seed", BigInteger, nullable=False),
    Column("max_steps", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("result", JSONB, nullable=True),
    Column("model_version", Text, nullable=True),
    Column("enqueued_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    CheckConstraint(f"status IN ('{PENDING}', '{DONE}')", name="tasks_status_check"),
)

# How many times the broker has delivered a task, redeliveries after a crashed worker included.
task_deliveries = Table(
    "task_deliveries",
    MetaData(),
    Column("task_id", PG_UUID(as_uuid=True), primary_key=True),
    Column("deliveries", Integer, nullable=False),
)

# Serialises concurrent first starts (API and worker): CREATE TABLE IF NOT EXISTS alone can
# still fail with a duplicate pg_type row when two sessions create the table at once.
_SCHEMA_LOCK_KEY = 7_301_045_001


@dataclass(frozen=True)
class TaskRecord:
    task_id: UUID
    seed: int
    max_steps: int
    status: str
    result: dict | None
    model_version: str | None
    enqueued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


async def ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(select(func.pg_advisory_xact_lock(_SCHEMA_LOCK_KEY)))
        await connection.execute(CreateTable(tasks, if_not_exists=True))
        await connection.execute(CreateTable(task_deliveries, if_not_exists=True))


async def insert_pending(engine: AsyncEngine, message: TaskMessage) -> None:
    """Record an accepted task; a no-op if the row exists (e.g. the worker already finished it)."""
    statement = (
        insert(tasks)
        .values(
            task_id=message.task_id,
            seed=message.seed,
            max_steps=message.max_steps,
            status=PENDING,
            enqueued_at=message.enqueued_at,
        )
        .on_conflict_do_nothing(index_elements=[tasks.c.task_id])
    )
    async with engine.begin() as connection:
        await connection.execute(statement)


async def save_result(
    engine: AsyncEngine,
    message: TaskMessage,
    result: dict,
    model_version: str,
    started_at: datetime,
    finished_at: datetime,
) -> bool:
    """Store the result once. Idempotent through the primary key: no read-then-write.

    Returns True if this call wrote the DONE row, False if the task was already DONE.
    """
    values = {
        "status": DONE,
        "result": result,
        "model_version": model_version,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    statement = (
        insert(tasks)
        .values(
            task_id=message.task_id,
            seed=message.seed,
            max_steps=message.max_steps,
            enqueued_at=message.enqueued_at,
            **values,
        )
        .on_conflict_do_update(
            index_elements=[tasks.c.task_id],
            set_=values,
            where=tasks.c.status == PENDING,
        )
    )
    async with engine.begin() as connection:
        outcome = await connection.execute(statement)
    return outcome.rowcount == 1


async def register_delivery(engine: AsyncEngine, task_id: UUID) -> int:
    """Count one more delivery of the task and return the new total (1 on the first delivery).

    One atomic upsert, so concurrent consumers and a restarted worker never lose a count.
    """
    statement = (
        insert(task_deliveries)
        .values(task_id=task_id, deliveries=1)
        .on_conflict_do_update(
            index_elements=[task_deliveries.c.task_id],
            set_={"deliveries": task_deliveries.c.deliveries + 1},
        )
        .returning(task_deliveries.c.deliveries)
    )
    async with engine.begin() as connection:
        return (await connection.execute(statement)).scalar_one()


async def get_task(engine: AsyncEngine, task_id: UUID) -> TaskRecord | None:
    async with engine.connect() as connection:
        row = (
            await connection.execute(select(tasks).where(tasks.c.task_id == task_id))
        ).one_or_none()
    return TaskRecord(**row._mapping) if row is not None else None
