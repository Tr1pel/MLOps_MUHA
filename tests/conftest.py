import io
import json
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_platform.main as main_module
from agent_platform.core.config import Settings
from agent_platform.core.logging import build_handler
from agent_platform.core.telemetry import build_views
from agent_platform.main import create_app
from agent_platform.ml.inference import LoadedModel, ModelInfo

# Providers are global and settable once per process, so the whole test session shares these;
# setup_telemetry() sees an SDK provider already installed and leaves it alone.
SPAN_EXPORTER = InMemorySpanExporter()
METRIC_READER = InMemoryMetricReader()
_tracer_provider = TracerProvider()
_tracer_provider.add_span_processor(SimpleSpanProcessor(SPAN_EXPORTER))
trace.set_tracer_provider(_tracer_provider)
metrics.set_meter_provider(
    MeterProvider(metric_readers=[METRIC_READER], views=build_views(Settings(_env_file=None)))
)
AsyncPGInstrumentor().instrument()


@pytest.fixture(autouse=True)
def spans():
    SPAN_EXPORTER.clear()
    return SPAN_EXPORTER


def metric_total(name: str, **attributes) -> float:
    """Sum of a counter's (or histogram's count) points whose attributes include ``attributes``."""
    data = METRIC_READER.get_metrics_data()
    total = 0.0
    for resource_metrics in data.resource_metrics if data else []:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    if attributes.items() <= dict(point.attributes).items():
                        total += getattr(point, "value", None) or getattr(point, "count", 0)
    return total


@pytest.fixture
def settings():
    return Settings(_env_file=None, app_commit="abc123", health_timeout_seconds=0.1)


class FakeBroker:
    """Stands in for RabbitBroker so API tests need neither RabbitMQ nor PostgreSQL."""

    def __init__(self, *_):
        self.start = AsyncMock()
        self.stop = AsyncMock()


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch):
    monkeypatch.setattr(main_module, "ensure_schema", AsyncMock())
    monkeypatch.setattr(main_module, "declare_topology", AsyncMock())


@pytest.fixture
def app(settings):
    return create_app(settings, broker_factory=FakeBroker)


@pytest.fixture
async def client(app):
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
def log_output():
    """Everything logged anywhere, as the JSON lines the services would print."""
    output = io.StringIO()
    handler = build_handler(output, service="test")
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield output
    root.removeHandler(handler)
    root.setLevel(previous_level)


def log_records(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


class FakePolicy:
    """Stands in for a fitted pipeline: always 'forward', counts predict calls."""

    def __init__(self, action: int = 0):
        self.action = action
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.full(len(X), self.action, dtype=np.int64)


def make_loaded(model=None, version="3", run_id="run-abc", name="fly-policy", alias="champion"):
    return LoadedModel(
        model=model if model is not None else FakePolicy(),
        info=ModelInfo(
            name=name,
            alias=alias,
            version=version,
            run_id=run_id,
            source="mlflow-artifacts:/1/abc/artifacts/model",
            loaded_at=datetime(2026, 1, 1, tzinfo=UTC),
            load_duration_ms=12.5,
        ),
    )
