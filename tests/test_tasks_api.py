import importlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from faststream.rabbit import TestRabbitBroker

import agent_platform.main as main_module
from agent_platform.core.config import Settings
from agent_platform.db.tasks import TaskRecord
from agent_platform.main import create_app
from agent_platform.messaging import (
    BrokerUnavailableError,
    QueueFullError,
    TaskMessage,
    make_broker,
)

tasks_module = importlib.import_module("agent_platform.api.tasks")
retry_after_seconds = tasks_module.retry_after_seconds


@pytest.fixture
def calls(monkeypatch):
    """Record the order of publish / insert; both succeed unless a test replaces them."""
    order = []

    async def publish(broker, settings, message):
        order.append(("publish", message))

    async def insert(engine, message):
        order.append(("insert", message))

    monkeypatch.setattr(tasks_module, "publish_task", publish)
    monkeypatch.setattr(tasks_module, "insert_pending", insert)
    return order


async def test_accepted_task_is_published_before_it_is_stored(client, calls):
    response = await client.post("/api/v1/tasks", json={"seed": 7, "max_steps": 20})
    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert response.json() == {"task_id": task_id, "status": "PENDING"}
    assert response.headers["location"] == f"/api/v1/tasks/{task_id}"
    assert [step for step, _ in calls] == ["publish", "insert"]
    message = calls[0][1]
    assert calls[1][1] is message
    assert message.task_id == UUID(task_id)
    assert (message.seed, message.max_steps) == (7, 20)


async def test_max_steps_has_a_default(client, calls):
    response = await client.post("/api/v1/tasks", json={"seed": 1})
    assert response.status_code == 202
    assert calls[0][1].max_steps == TaskMessage.model_fields["max_steps"].default


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (QueueFullError("tasks"), "queue_full"),
        (BrokerUnavailableError("down"), "broker_unavailable"),
    ],
)
async def test_refused_publish_is_503_with_retry_after_and_nothing_stored(
    client, monkeypatch, error, code
):
    insert = AsyncMock()
    monkeypatch.setattr(tasks_module, "publish_task", AsyncMock(side_effect=error))
    monkeypatch.setattr(tasks_module, "insert_pending", insert)
    response = await client.post("/api/v1/tasks", json={"seed": 1})
    assert response.status_code == 503
    assert response.json() == {"error": code}
    assert response.headers["retry-after"] == "13"  # ceil(50 * 0.5 s / 2 workers)
    assert "location" not in response.headers
    insert.assert_not_awaited()


async def test_failed_insert_after_publish_still_accepts_the_task(client, monkeypatch, log_output):
    monkeypatch.setattr(tasks_module, "publish_task", AsyncMock())
    monkeypatch.setattr(tasks_module, "insert_pending", AsyncMock(side_effect=OSError("db down")))
    response = await client.post("/api/v1/tasks", json={"seed": 1})
    assert response.status_code == 202
    assert "pending_insert_failed" in log_output.getvalue()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, 13),  # defaults: 50 * 0.5 / 2 = 12.5
        ({"queue_max_length": 100, "inference_seconds": 2.0, "worker_concurrency": 4}, 50),
        ({"queue_max_length": 10, "inference_seconds": 0.5, "worker_concurrency": 5}, 1),
        ({"queue_max_length": 1, "inference_seconds": 0.0}, 1),
        ({"queue_max_length": 0, "inference_seconds": 3.0, "worker_concurrency": 2}, 2),
        ({"queue_max_length": 0, "inference_seconds": 0.1}, 1),
    ],
)
def test_retry_after_is_the_time_to_drain_a_full_queue(values, expected):
    assert retry_after_seconds(Settings(_env_file=None, **values)) == expected


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": "x"},
        {"seed": 1, "max_steps": 0},
        {"seed": 1, "max_steps": 1001},
        {"seed": 1, "extra": 1},
    ],
)
async def test_invalid_body_is_422_and_not_published(client, calls, body):
    response = await client.post("/api/v1/tasks", json=body)
    assert response.status_code == 422
    assert calls == []


def make_record(**changes):
    values = {
        "task_id": uuid4(),
        "seed": 3,
        "max_steps": 10,
        "status": "PENDING",
        "result": None,
        "model_version": None,
        "enqueued_at": datetime(2026, 1, 1, tzinfo=UTC),
        "started_at": None,
        "finished_at": None,
    }
    return TaskRecord(**{**values, **changes})


async def test_get_done_task(client, monkeypatch):
    finished = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    record = make_record(
        status="DONE",
        result={"seed": 3, "success": True, "steps": 4},
        model_version="5",
        started_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        finished_at=finished,
    )
    monkeypatch.setattr(tasks_module, "get_task", AsyncMock(return_value=record))
    response = await client.get(f"/api/v1/tasks/{record.task_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DONE"
    assert body["task_id"] == str(record.task_id)
    assert body["model_version"] == "5"
    assert body["result"] == {"seed": 3, "success": True, "steps": 4}
    assert datetime.fromisoformat(body["finished_at"]) == finished


async def test_get_pending_task_has_null_result(client, monkeypatch):
    record = make_record()
    monkeypatch.setattr(tasks_module, "get_task", AsyncMock(return_value=record))
    body = (await client.get(f"/api/v1/tasks/{record.task_id}")).json()
    assert body["status"] == "PENDING"
    assert body["result"] is None and body["started_at"] is None


async def test_get_unknown_task_is_404(client, monkeypatch):
    monkeypatch.setattr(tasks_module, "get_task", AsyncMock(return_value=None))
    response = await client.get(f"/api/v1/tasks/{uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"error": "not_found"}


async def test_get_with_malformed_id_is_422(client):
    assert (await client.get("/api/v1/tasks/not-a-uuid")).status_code == 422


async def test_post_puts_the_message_into_the_tasks_queue(monkeypatch):
    settings = Settings(_env_file=None, tasks_queue="jobs-test")
    received = []
    inserted = AsyncMock()
    monkeypatch.setattr(tasks_module, "insert_pending", inserted)
    monkeypatch.setattr(main_module, "create_engine", lambda s: AsyncMock())
    broker = make_broker(settings)

    @broker.subscriber(settings.tasks_queue)
    async def collect(body: dict):
        received.append(body)

    app = create_app(settings, broker_factory=lambda s: broker)
    async with TestRabbitBroker(broker):
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/api/v1/tasks", json={"seed": 11, "max_steps": 5})
    assert response.status_code == 202
    [body] = received
    assert body["task_id"] == response.json()["task_id"]
    assert (body["seed"], body["max_steps"]) == (11, 5)
    inserted.assert_awaited_once()
