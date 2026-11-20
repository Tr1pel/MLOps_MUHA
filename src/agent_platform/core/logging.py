"""Structured JSON logging for every process of the platform (API, worker, source).

Everything goes through the standard library ``logging``: uvicorn, FastStream/aio-pika,
SQLAlchemy, MLflow and ``warnings`` all write to it, so one JSON handler on the *root* logger
gives their records the same format without bridges and without new dependencies. The handler
(not a logger) carries the filters, because logger filters do not see records propagated from
child loggers.

One line on stdout is one ``json.dumps`` object. Fields: ``timestamp`` (UTC, ISO 8601),
``level``, ``logger``, ``event``, ``service`` always; ``task_id``, ``trace_id``, ``span_id``,
``request_id``, ``exception`` and any ``extra=`` fields when present. Log call sites pass facts
as ``extra=`` fields, never inside the event text, and never secrets, message bodies or records.
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from opentelemetry import trace
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agent_platform.core.telemetry import resolve_service_name

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
# The task being handled by this asyncio task / thread; copied into pool threads explicitly.
task_id_context: ContextVar[str | None] = ContextVar("task_id", default=None)
logger = logging.getLogger("agent_platform")

# Liveness probes fire every few seconds; they are not worth an INFO line each.
QUIET_PATHS = frozenset({"/healthz"})
EXPORTER_LOG_INTERVAL_SECONDS = 60.0

_STANDARD_ATTRIBUTES = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


class ContextFilter(logging.Filter):
    """Add the current trace/span ids, task id and request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            record.trace_id = format(context.trace_id, "032x")
            record.span_id = format(context.span_id, "016x")
        task_id = task_id_context.get()
        if task_id is not None and not hasattr(record, "task_id"):
            record.task_id = task_id
        request_id = request_id_context.get()
        if request_id is not None:
            record.request_id = request_id
        return True


class ExporterNoiseFilter(logging.Filter):
    """Let an OpenTelemetry export failure through once a minute per logger.

    With the collector down, the exporters complain on every export interval. The first message
    keeps the problem visible; the repeats carry no new information and would bury real logs.
    """

    def __init__(self, interval: float = EXPORTER_LOG_INTERVAL_SECONDS) -> None:
        super().__init__()
        self._interval = interval
        self._last: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING or not record.name.startswith("opentelemetry."):
            return True
        now = time.monotonic()
        last = self._last.get(record.name)
        if last is not None and now - last < self._interval:
            return False
        self._last[record.name] = now
        return True


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str = "agent_platform") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "service": self.service,
        }
        for field in ("task_id", "trace_id", "span_id", "request_id"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        for field, value in record.__dict__.items():
            if field not in _STANDARD_ATTRIBUTES and field not in payload:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str: a UUID or datetime in extra= must not turn into a "Logging error".
        return json.dumps(payload, ensure_ascii=False, default=str)


def build_handler(stream=None, service: str = "agent_platform") -> logging.Handler:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    handler.addFilter(ContextFilter())
    handler.addFilter(ExporterNoiseFilter())
    handler._agent_platform = True  # type: ignore[attr-defined]
    return handler


def _log_uncaught(exc_type, exc, traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, traceback)
        return
    logger.critical("uncaught_exception", exc_info=(exc_type, exc, traceback))


def configure_logging(level: str, service_default: str = "agent_platform") -> None:
    """Install the JSON handler on the root logger; safe to call repeatedly."""
    root = logging.getLogger()
    # Replace only our own handler: pytest's capture handlers must survive.
    root.handlers = [h for h in root.handlers if not getattr(h, "_agent_platform", False)]
    root.addHandler(build_handler(service=resolve_service_name(service_default)))
    root.setLevel(level)
    # uvicorn installs its own text handlers; hand its loggers over to the root handler. An access
    # logger without handlers is one uvicorn switched off (--no-access-log): leave it silent.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        library_logger = logging.getLogger(name)
        if name == "uvicorn.access" and not library_logger.handlers:
            continue
        library_logger.handlers = []
        library_logger.propagate = True
        library_logger.setLevel(logging.NOTSET)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught


class RequestLoggingMiddleware:
    """Log completed HTTP requests without buffering bodies or blocking streaming."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        token = request_id_context.set(request_id)
        # A handler may set the task id; it must not outlive the request in the caller's context.
        task_token = task_id_context.set(None)
        started = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.log(
                logging.DEBUG if scope["path"] in QUIET_PATHS else logging.INFO,
                "http_request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
            task_id_context.reset(task_token)
            request_id_context.reset(token)
