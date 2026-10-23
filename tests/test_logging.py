import json
import logging

import pytest

from agent_platform.core.logging import (
    JsonFormatter,
    RequestLoggingMiddleware,
    request_id_context,
)


async def test_request_log_contains_metadata_but_no_query_secrets(client, log_output):
    response = await client.get("/healthz?token=do-not-log-this")
    records = [json.loads(line) for line in log_output.getvalue().splitlines()]
    record = next(record for record in records if record["event"] == "http_request")
    assert record["request_id"] == response.headers["x-request-id"]
    assert record["method"] == "GET"
    assert record["path"] == "/healthz"
    assert record["status_code"] == 200
    assert record["duration_ms"] >= 0
    assert "do-not-log-this" not in json.dumps(records)
    assert request_id_context.get() is None


def test_formatter_escapes_multiline_messages():
    record = logging.LogRecord("test", logging.INFO, "", 0, "first\nsecond", (), None)
    output = JsonFormatter().format(record)
    assert "\n" not in output
    assert json.loads(output)["event"] == "first\nsecond"


async def test_middleware_passes_non_http_scopes_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope)

    middleware = RequestLoggingMiddleware(inner)
    scope = {"type": "lifespan"}
    await middleware(scope, None, None)
    assert seen == [scope]


async def test_middleware_resets_context_after_exception():
    async def inner(scope, receive, send):
        assert request_id_context.get()
        raise RuntimeError("handler failed")

    middleware = RequestLoggingMiddleware(inner)
    with pytest.raises(RuntimeError, match="handler failed"):
        await middleware({"type": "http", "method": "GET", "path": "/error"}, None, None)
    assert request_id_context.get() is None
