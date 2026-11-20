import io
import json
import logging
import sys
import time
import warnings
from contextvars import copy_context
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from conftest import log_records
from opentelemetry import trace

from agent_platform.core.logging import (
    ExporterNoiseFilter,
    JsonFormatter,
    RequestLoggingMiddleware,
    build_handler,
    configure_logging,
    request_id_context,
    task_id_context,
)

REQUIRED = {"timestamp", "level", "logger", "event", "service"}


def emit(output: io.StringIO, name="agent_platform", level=logging.INFO, msg="hello", **kwargs):
    logging.getLogger(name).log(level, msg, **kwargs)
    return log_records(output)[-1]


async def test_request_log_contains_metadata_but_no_query_secrets(client, log_output):
    response = await client.get("/api/v1/version?token=do-not-log-this")
    records = log_records(log_output)
    record = next(record for record in records if record["event"] == "http_request")
    assert record["request_id"] == response.headers["x-request-id"]
    assert record["method"] == "GET"
    assert record["path"] == "/api/v1/version"
    assert record["status_code"] == 200
    assert record["duration_ms"] >= 0
    assert record["logger"] == "agent_platform"
    assert "do-not-log-this" not in json.dumps([r for r in records if r["logger"] != "httpx"])
    assert request_id_context.get() is None


async def test_liveness_probe_is_logged_at_debug_not_info(client, log_output):
    await client.get("/healthz")
    assert not [r for r in log_records(log_output) if r["event"] == "http_request"]

    logging.getLogger().setLevel(logging.DEBUG)
    await client.get("/healthz")
    [record] = [r for r in log_records(log_output) if r["event"] == "http_request"]
    assert (record["level"], record["path"]) == ("DEBUG", "/healthz")


def test_record_has_the_contract_fields(log_output):
    record = emit(log_output, msg="task_done", extra={"written": True, "delivery": 2})
    assert REQUIRED <= record.keys()
    assert record["level"] == "INFO"
    assert record["event"] == "task_done"
    assert record["service"] == "test"
    assert (record["written"], record["delivery"]) == (True, 2)
    assert datetime.fromisoformat(record["timestamp"]).utcoffset().total_seconds() == 0
    # Outside a span, a task or a request, those fields are absent rather than null.
    assert not {"task_id", "trace_id", "span_id", "request_id", "exception"} & record.keys()


def test_line_uses_default_json_separators_and_keeps_unicode(log_output):
    emit(log_output, msg="задача")
    line = log_output.getvalue().splitlines()[-1]
    assert '"event": "задача"' in line


def test_extra_values_that_json_cannot_encode_do_not_break_logging(log_output):
    task_id, moment = uuid4(), datetime(2026, 1, 1, tzinfo=UTC)
    record = emit(log_output, extra={"id": task_id, "at": moment})
    assert UUID(record["id"]) == task_id
    assert record["at"].startswith("2026-01-01")


def test_task_id_comes_from_the_context_var_and_an_explicit_extra_wins(log_output):
    token = task_id_context.set("11111111-1111-1111-1111-111111111111")
    try:
        assert emit(log_output)["task_id"] == "11111111-1111-1111-1111-111111111111"
        assert emit(log_output, extra={"task_id": "other"})["task_id"] == "other"
    finally:
        task_id_context.reset(token)
    assert "task_id" not in emit(log_output)


def test_trace_and_span_ids_come_from_the_current_span(log_output):
    with trace.get_tracer("t").start_as_current_span("work") as span:
        record = emit(log_output)
        context = span.get_span_context()
    assert record["trace_id"] == format(context.trace_id, "032x")
    assert record["span_id"] == format(context.span_id, "016x")
    assert len(record["trace_id"]) == 32 and len(record["span_id"]) == 16


def test_records_from_library_loggers_get_the_same_format_and_context(log_output):
    token = task_id_context.set("abc")
    try:
        with trace.get_tracer("t").start_as_current_span("work"):
            record = emit(log_output, name="aio_pika.connection", msg="library line")
    finally:
        task_id_context.reset(token)
    assert REQUIRED <= record.keys()
    assert record["logger"] == "aio_pika.connection"
    assert record["task_id"] == "abc" and "trace_id" in record


def test_exception_is_a_full_traceback_string(log_output):
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("agent_platform").error("task_failed", exc_info=True)
    record = log_records(log_output)[-1]
    assert record["exception"].startswith("Traceback (most recent call last)")
    assert 'raise ValueError("boom")' in record["exception"]
    assert record["exception"].rstrip().endswith("ValueError: boom")


def test_formatter_escapes_multiline_messages():
    record = logging.LogRecord("test", logging.INFO, "", 0, "first\nsecond", (), None)
    output = JsonFormatter().format(record)
    assert "\n" not in output
    assert json.loads(output)["event"] == "first\nsecond"


def test_context_survives_a_copied_context_in_another_thread(log_output):
    from concurrent.futures import ThreadPoolExecutor

    token = task_id_context.set("from-handler")
    try:
        with ThreadPoolExecutor(1) as pool:
            pool.submit(copy_context().run, logging.getLogger("t").info, "in thread").result()
    finally:
        task_id_context.reset(token)
    assert log_records(log_output)[-1]["task_id"] == "from-handler"


def test_configure_logging_routes_uvicorn_and_warnings_through_the_json_handler(capfd):
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, sys.excepthook, warnings.showwarning)
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.handlers = [logging.StreamHandler()]
    uvicorn_logger.propagate = False
    access = logging.getLogger("uvicorn.access")
    access.handlers, access.propagate = [], False  # what --no-access-log leaves behind
    logging.captureWarnings(False)  # pytest's own warning capture replaced showwarning
    try:
        configure_logging("INFO", "worker")
        configure_logging("INFO", "worker")  # idempotent: still exactly one of our handlers
        assert len([h for h in root.handlers if getattr(h, "_agent_platform", False)]) == 1
        logging.getLogger("uvicorn.error").info("Started server process")
        logging.getLogger("uvicorn.access").info("GET /healthz")
        warnings.warn("careful", UserWarning, stacklevel=1)
        out = [json.loads(line) for line in capfd.readouterr().out.splitlines()]
    finally:
        logging.captureWarnings(False)
        root.handlers, root.level = saved[0], saved[1]
        sys.excepthook, warnings.showwarning = saved[2], saved[3]
        uvicorn_logger.propagate = True
        access.propagate = True
    by_logger = {r["logger"]: r for r in out}
    assert by_logger["uvicorn.error"]["event"] == "Started server process"
    assert by_logger["uvicorn.error"]["service"] == "worker"
    assert "careful" in by_logger["py.warnings"]["event"]
    assert "uvicorn.access" not in by_logger


def test_service_name_prefers_otel_service_name(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env")
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, sys.excepthook)
    try:
        configure_logging("INFO", "default-name")
        formatter = next(h for h in root.handlers if getattr(h, "_agent_platform", False)).formatter
        assert formatter.service == "from-env"
    finally:
        logging.captureWarnings(False)
        root.handlers, root.level, sys.excepthook = saved


def test_uncaught_exception_hook_logs_a_json_error_with_traceback(log_output):
    from agent_platform.core.logging import _log_uncaught

    try:
        raise RuntimeError("no model")
    except RuntimeError:
        _log_uncaught(*sys.exc_info())
    record = log_records(log_output)[-1]
    assert (record["level"], record["event"]) == ("CRITICAL", "uncaught_exception")
    assert "RuntimeError: no model" in record["exception"]


def test_keyboard_interrupt_keeps_the_default_hook(monkeypatch):
    from agent_platform.core.logging import _log_uncaught

    seen = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *args: seen.append(args[0]))
    _log_uncaught(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert seen == [KeyboardInterrupt]


def test_exporter_failures_are_logged_once_per_interval():
    noise = ExporterNoiseFilter(interval=0.2)

    def record(name="opentelemetry.exporter.otlp.proto.grpc.exporter", level=logging.WARNING):
        return logging.LogRecord(name, level, "", 0, "Failed to export", (), None)

    assert noise.filter(record()) is True
    assert noise.filter(record()) is False
    assert noise.filter(record("opentelemetry.sdk.metrics._internal.export")) is True
    assert noise.filter(record(level=logging.INFO)) is True  # only failures are throttled
    assert noise.filter(record("agent_platform")) is True
    time.sleep(0.25)
    assert noise.filter(record()) is True


def test_noise_filter_is_installed_on_the_handler():
    handler = build_handler(io.StringIO())
    assert any(isinstance(f, ExporterNoiseFilter) for f in handler.filters)


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
        task_id_context.set("set-by-handler")
        raise RuntimeError("handler failed")

    middleware = RequestLoggingMiddleware(inner)
    with pytest.raises(RuntimeError, match="handler failed"):
        await middleware({"type": "http", "method": "GET", "path": "/error"}, None, None)
    assert request_id_context.get() is None
    assert task_id_context.get() is None
