import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
from conftest import METRIC_READER
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from agent_platform.core import telemetry
from agent_platform.core.config import Settings
from agent_platform.core.telemetry import (
    build_providers,
    build_resource,
    build_views,
    inference_buckets,
    resolve_service_name,
    setup_telemetry,
    shutdown_telemetry,
)

OTEL_VARIABLES = (
    "OTEL_SERVICE_NAME",
    "OTEL_RESOURCE_ATTRIBUTES",
    "OTEL_SDK_DISABLED",
    "OTEL_TRACES_SAMPLER",
    "OTEL_TRACES_SAMPLER_ARG",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in OTEL_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_resource_defaults_to_given_service_name_and_hostname_as_instance_id():
    attributes = build_resource("worker").attributes
    assert attributes["service.name"] == "worker"
    assert attributes["service.instance.id"] == socket.gethostname()


def test_resource_takes_everything_it_can_from_the_environment(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "api-from-env")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.instance.id=replica-7,service.version=1.2.3,deployment.environment.name=dev",
    )
    attributes = build_resource("worker").attributes
    assert attributes["service.name"] == "api-from-env"
    assert attributes["service.instance.id"] == "replica-7"
    assert attributes["service.version"] == "1.2.3"
    assert attributes["deployment.environment.name"] == "dev"


def test_service_name_can_come_from_resource_attributes(monkeypatch):
    assert resolve_service_name("fallback") == "fallback"
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=from-attrs")
    assert resolve_service_name("fallback") == "from-attrs"
    assert build_resource("fallback").attributes["service.name"] == "from-attrs"


def test_inference_buckets_scale_with_the_target_duration():
    assert inference_buckets(0.5) == [0.0, 0.5, 0.505, 0.51, 0.525, 0.55, 0.6, 0.75, 1.0, 2.0]
    assert inference_buckets(5.0)[1] == 5.0
    # A zero target (used in tests) must still give valid, sorted, duplicate-free boundaries.
    assert inference_buckets(0.0) == [0.0]


def histogram_bounds(reader, name):
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    return list(metric.data.data_points[0].explicit_bounds)
    raise AssertionError(f"{name} not exported")


def test_views_give_histograms_the_contract_buckets():
    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader], views=build_views(Settings(_env_file=None, inference_seconds=0.5))
    )
    meter = provider.get_meter("t")
    for name in (
        "worker.inference.duration",
        "worker.task.duration",
        "http.server.request.duration",
    ):
        meter.create_histogram(name, unit="s").record(0.3)

    assert histogram_bounds(reader, "worker.inference.duration") == inference_buckets(0.5)
    assert histogram_bounds(reader, "worker.task.duration") == [
        0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300
    ]  # fmt: skip
    assert histogram_bounds(reader, "http.server.request.duration") == [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5
    ]  # fmt: skip


def test_session_meter_provider_uses_the_same_views():
    meter = metrics.get_meter("t")
    meter.create_histogram("worker.inference.duration", unit="s").record(0.5)
    assert histogram_bounds(METRIC_READER, "worker.inference.duration") == inference_buckets(0.5)


def test_sampler_comes_from_the_standard_environment(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.25")
    tracer_provider, meter_provider = build_providers("api", Settings(_env_file=None))
    try:
        sampler = tracer_provider.sampler
        assert isinstance(sampler, ParentBased)
        assert isinstance(sampler._root, TraceIdRatioBased)
        assert sampler._root.rate == 0.25
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()


def test_setup_is_a_noop_when_the_sdk_is_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    build = MagicMock()
    monkeypatch.setattr(telemetry, "build_providers", build)
    assert setup_telemetry("api", Settings(_env_file=None)) is False
    build.assert_not_called()


def test_setup_leaves_an_installed_sdk_provider_alone(monkeypatch):
    # conftest installed one for the whole session.
    assert isinstance(trace.get_tracer_provider(), TracerProvider)
    build = MagicMock()
    monkeypatch.setattr(telemetry, "build_providers", build)
    assert setup_telemetry("api", Settings(_env_file=None)) is False
    build.assert_not_called()


def test_setup_installs_providers_and_instrumentation_then_shutdown_stops_them(monkeypatch):
    tracer_provider, meter_provider = MagicMock(), MagicMock()
    monkeypatch.setattr(telemetry, "build_providers", lambda *a: (tracer_provider, meter_provider))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: object())
    installed = []
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: installed.append(("trace", p)))
    monkeypatch.setattr(metrics, "set_meter_provider", lambda p: installed.append(("meter", p)))
    instrumented = MagicMock()
    monkeypatch.setattr(telemetry, "instrument_libraries", instrumented)

    assert setup_telemetry("worker", Settings(_env_file=None)) is True
    assert installed == [("trace", tracer_provider), ("meter", meter_provider)]
    instrumented.assert_called_once()

    shutdown_telemetry()
    tracer_provider.shutdown.assert_called_once()
    meter_provider.shutdown.assert_called_once()
    shutdown_telemetry()  # second call is a no-op
    tracer_provider.shutdown.assert_called_once()


def test_shutdown_does_not_wait_for_a_hung_provider(monkeypatch):
    release = threading.Event()
    hung = MagicMock()
    hung.shutdown.side_effect = lambda: release.wait(10)
    fast = MagicMock()
    monkeypatch.setattr(telemetry, "_providers", (hung, fast))
    monkeypatch.setattr(telemetry, "SHUTDOWN_JOIN_SECONDS", 0.2)
    started = time.perf_counter()
    shutdown_telemetry()
    release.set()
    assert time.perf_counter() - started < 1
    fast.shutdown.assert_called_once()


def test_unreachable_collector_neither_raises_nor_blocks_for_long(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.invalid:4317")
    tracer_provider, meter_provider = build_providers("api", Settings(_env_file=None))
    with tracer_provider.get_tracer("t").start_as_current_span("work"):
        pass
    meter_provider.get_meter("t").create_counter("c").add(1)
    monkeypatch.setattr(telemetry, "_providers", (tracer_provider, meter_provider))

    started = time.perf_counter()
    shutdown_telemetry()
    assert time.perf_counter() - started < telemetry.SHUTDOWN_JOIN_SECONDS * 2 + 1


def test_http_semantic_conventions_opt_in_is_set_before_instrumentation():
    import os

    assert os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] == "http"


def test_asyncpg_connection_methods_are_instrumented():
    import asyncpg

    # A span per statement against a real server is checked in the integration tests.
    assert hasattr(asyncpg.Connection.execute, "__wrapped__")
    assert hasattr(asyncpg.Connection.fetch, "__wrapped__")
