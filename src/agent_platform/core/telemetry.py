"""OpenTelemetry setup: traces and metrics leave the process over OTLP, logs stay on stdout.

The application knows nothing about storage. Where the collector is, how to sample and which
resource attributes to add come from the standard ``OTEL_*`` environment variables read by the
SDK; this module only adds what the SDK does not do: a readable ``service.instance.id``, the
histogram buckets, bounded export/shutdown times and an unreachable collector that never
breaks the service.
"""

import os
import socket
import threading
from collections.abc import Callable, Iterable

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import OTELResourceDetector, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent_platform.core.config import Settings

# Without this the ASGI instrumentation reports the old `http.server.duration` in milliseconds.
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http")

# The SDK default is 10 s per export and up to ~17 s per flush with the collector down, which
# would make the one-shot source and SIGTERM of the worker hang. 1 s is plenty inside a
# docker network.
EXPORT_TIMEOUT_SECONDS = 1.0
SHUTDOWN_JOIN_SECONDS = EXPORT_TIMEOUT_SECONDS + 1.0

INFERENCE_BUCKET_FACTORS = (1, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2, 4)
TASK_DURATION_BUCKETS = (0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300)
HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)

_providers: tuple[TracerProvider, MeterProvider] | None = None


def sdk_disabled() -> bool:
    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true"


def resolve_service_name(default: str) -> str:
    """OTEL_SERVICE_NAME, else ``service.name`` from OTEL_RESOURCE_ATTRIBUTES, else the default."""
    return str(OTELResourceDetector().detect().attributes.get("service.name") or default)


def build_resource(service_name_default: str) -> Resource:
    """Resource from the environment plus defaults for what the environment leaves unset.

    Explicit attributes passed to ``Resource.create`` would win over the environment, so the
    defaults are merged underneath instead.
    """
    from_env = OTELResourceDetector().detect().attributes
    defaults: dict[str, str] = {}
    if "service.name" not in from_env:
        defaults["service.name"] = service_name_default
    if "service.instance.id" not in from_env:
        # The SDK has a detector for it, but it yields a random UUID that is new on every
        # restart. Replicas started with `--scale` share one environment and differ in their
        # hostname (the container id), which is also what `docker ps` shows.
        defaults["service.instance.id"] = socket.gethostname()
    return Resource.create().merge(Resource(defaults))


def inference_buckets(inference_seconds: float) -> list[float]:
    """Boundaries around the target inference time (the worker pads every task to it)."""
    return sorted(
        {0.0, *(round(inference_seconds * factor, 6) for factor in INFERENCE_BUCKET_FACTORS)}
    )


def build_views(settings: Settings) -> list[View]:
    def histogram(name: str, boundaries: Iterable[float]) -> View:
        return View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(sorted(boundaries)),
        )

    return [
        histogram("worker.inference.duration", inference_buckets(settings.inference_seconds)),
        histogram("worker.task.duration", TASK_DURATION_BUCKETS),
        histogram("http.server.request.duration", HTTP_DURATION_BUCKETS),
    ]


def build_providers(
    service_name_default: str, settings: Settings
) -> tuple[TracerProvider, MeterProvider]:
    """Providers exporting over OTLP/gRPC; sampler and endpoint come from OTEL_* variables."""
    resource = build_resource(service_name_default)
    export_millis = int(EXPORT_TIMEOUT_SECONDS * 1000)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(timeout=EXPORT_TIMEOUT_SECONDS), export_timeout_millis=export_millis
        )
    )
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(timeout=EXPORT_TIMEOUT_SECONDS),
        # None: take the interval from OTEL_METRIC_EXPORT_INTERVAL.
        export_interval_millis=None,
        export_timeout_millis=export_millis,
    )
    meter_provider = MeterProvider(
        resource=resource, metric_readers=[reader], views=build_views(settings)
    )
    return tracer_provider, meter_provider


def instrument_libraries() -> None:
    """Process memory metrics and a span per statement sent through asyncpg.

    asyncpg rather than SQLAlchemy instrumentation: it does not depend on the SQLAlchemy
    version (the SQLAlchemy one refuses to work with 2.1) and patches the driver once for
    every engine, whenever it is created.
    """
    SystemMetricsInstrumentor(
        config={"process.memory.usage": None, "process.memory.virtual": None}
    ).instrument()
    AsyncPGInstrumentor().instrument()


def setup_telemetry(service_name_default: str, settings: Settings) -> bool:
    """Install the global providers. Returns False (and does nothing) if disabled or already set."""
    global _providers
    if sdk_disabled() or isinstance(trace.get_tracer_provider(), TracerProvider):
        return False
    tracer_provider, meter_provider = build_providers(service_name_default, settings)
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    _providers = (tracer_provider, meter_provider)
    instrument_libraries()
    return True


def shutdown_telemetry() -> None:
    """Export what is buffered and stop, in at most ~``SHUTDOWN_JOIN_SECONDS`` even with no collector."""
    global _providers
    providers, _providers = _providers, None
    if providers is None:
        return
    shutdowns: list[Callable[[], None]] = [provider.shutdown for provider in providers]
    threads = [threading.Thread(target=shutdown, daemon=True) for shutdown in shutdowns]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(SHUTDOWN_JOIN_SECONDS)


def instrument_fastapi(app: FastAPI) -> None:
    FastAPIInstrumentor.instrument_app(
        app, excluded_urls="healthz", exclude_spans=["receive", "send"]
    )
