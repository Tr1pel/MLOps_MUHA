import importlib
from unittest.mock import AsyncMock

import pytest
from conftest import log_records, metric_total
from opentelemetry.trace import SpanKind, StatusCode

from agent_platform.messaging import BrokerUnavailableError, QueueFullError

tasks_module = importlib.import_module("agent_platform.api.tasks")


def server_spans(spans):
    return [s for s in spans.get_finished_spans() if s.kind is SpanKind.SERVER]


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (QueueFullError("tasks"), "queue_full"),
        (BrokerUnavailableError("down"), "broker_unavailable"),
    ],
)
async def test_503_is_an_error_span_with_the_exception_and_a_warning(
    client, monkeypatch, spans, log_output, error, reason
):
    monkeypatch.setattr(tasks_module, "publish_task", AsyncMock(side_effect=error))
    response = await client.post("/api/v1/tasks", json={"seed": 1})
    assert response.status_code == 503

    [span] = server_spans(spans)
    assert span.name == "POST /api/v1/tasks"
    assert span.status.status_code is StatusCode.ERROR
    [event] = [e for e in span.events if e.name == "exception"]
    assert event.attributes["exception.type"].endswith(type(error).__name__)
    assert span.attributes["task.id"]

    [warning] = [r for r in log_records(log_output) if r["event"] == "task_rejected"]
    assert warning["level"] == "WARNING"
    assert (warning["reason"], warning["retry_after"]) == (reason, 13)
    assert warning["task_id"] == span.attributes["task.id"]
    assert warning["trace_id"] == format(span.context.trace_id, "032x")


async def test_accepted_task_is_an_ok_server_span_with_task_id(client, monkeypatch, spans):
    monkeypatch.setattr(tasks_module, "publish_task", AsyncMock())
    monkeypatch.setattr(tasks_module, "insert_pending", AsyncMock())
    response = await client.post("/api/v1/tasks", json={"seed": 1})
    [span] = server_spans(spans)
    assert span.status.status_code is not StatusCode.ERROR
    assert span.attributes["task.id"] == response.json()["task_id"]
    assert span.attributes["http.route"] == "/api/v1/tasks"


async def test_http_request_duration_metric_has_the_contract_name_and_attributes(client):
    attributes = {
        "http.route": "/api/v1/version",
        "http.request.method": "GET",
        "http.response.status_code": 200,
    }
    before = metric_total("http.server.request.duration", **attributes)
    await client.get("/api/v1/version")
    assert metric_total("http.server.request.duration", **attributes) == before + 1
    assert metric_total("http.server.duration") == 0  # not the old millisecond metric


async def test_liveness_probe_is_neither_traced_nor_measured(client, spans):
    await client.get("/healthz")
    assert server_spans(spans) == []
    assert metric_total("http.server.request.duration", **{"http.route": "/healthz"}) == 0


async def test_request_log_carries_the_trace_id_of_the_server_span(client, spans, log_output):
    await client.get("/api/v1/version")
    [span] = server_spans(spans)
    [record] = [r for r in log_records(log_output) if r["event"] == "http_request"]
    assert record["trace_id"] == format(span.context.trace_id, "032x")
