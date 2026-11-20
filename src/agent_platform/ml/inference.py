"""Load the registered policy from MLflow and run one episode with it (used by the worker)."""

import asyncio
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from sklearn import config_context

from agent_platform.core.config import Settings
from agent_platform.sim.env import OBS_DIM
from agent_platform.sim.episode import EpisodeResult, run_episode

NOT_CONFIGURED = "not_configured"
ALIAS_NOT_FOUND = "alias_not_found"
LOAD_TIMEOUT = "timeout"
LOAD_FAILED = "load_failed"


class ModelLoadError(Exception):
    """A load failure with a stable, secret-free category in ``reason``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ModelInfo:
    name: str
    alias: str
    version: str
    run_id: str | None
    source: str | None
    loaded_at: datetime
    load_duration_ms: float


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    info: ModelInfo


ModelLoader = Callable[[Settings], LoadedModel]


def _redact(uri: str | None) -> str | None:
    """Drop ``user:password@`` from a URI so it is safe to log and return."""
    if not uri:
        return uri
    parts = urlsplit(uri)
    if "@" not in parts.netloc:
        return uri
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _make_client(tracking_uri: str):
    # Imported lazily: mlflow import is slow and not needed when no model is configured.
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _load_sklearn(tracking_uri: str, model_uri: str, dst_path: str):
    import mlflow.sklearn

    import mlflow

    # models:/ URIs are resolved against the process-wide registry URI.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return mlflow.sklearn.load_model(model_uri, dst_path=dst_path)


def _is_missing_alias(exc: Exception) -> bool:
    from mlflow.exceptions import MlflowException

    return isinstance(exc, MlflowException) and exc.error_code == "RESOURCE_DOES_NOT_EXIST"


def load_registered_model(settings: Settings) -> LoadedModel:
    """Blocking: resolve the alias to a version, then load exactly that version."""
    uri = settings.mlflow_tracking_uri
    if uri is None:
        raise ModelLoadError(NOT_CONFIGURED)
    started = perf_counter()
    # MLflow's defaults (7 retries, 120 s timeout) would keep an unreachable server busy far
    # longer than the startup deadline; an operator can still override both in the environment.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    client = _make_client(uri)
    try:
        version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    except Exception as exc:
        if _is_missing_alias(exc):
            raise ModelLoadError(ALIAS_NOT_FOUND) from exc
        raise
    # Loading by the resolved number, not by alias, means the metadata reported by the API
    # can never disagree with the weights even if the alias is moved during the download.
    model_uri = f"models:/{settings.model_name}/{version.version}"
    with tempfile.TemporaryDirectory(prefix="model-") as scratch:
        model = _load_sklearn(uri, model_uri, scratch)
    # A wrong feature layout should fail startup, not the first user request.
    probe = np.asarray(model.predict(np.zeros((1, OBS_DIM))))
    if probe.shape != (1,):
        raise ModelLoadError(LOAD_FAILED)
    return LoadedModel(
        model=model,
        info=ModelInfo(
            name=settings.model_name,
            alias=settings.model_alias,
            version=str(version.version),
            run_id=version.run_id or None,
            source=_redact(version.source),
            loaded_at=datetime.now(UTC),
            load_duration_ms=round((perf_counter() - started) * 1000, 3),
        ),
    )


async def _run_in_daemon_thread[T](func: Callable[[], T], timeout_seconds: float) -> T:
    """Run blocking ``func`` off the loop; on timeout stop waiting and abandon the thread.

    A daemon thread (not ``asyncio.to_thread``) is used because a hung MLflow call cannot be
    interrupted and must not keep the process from shutting down.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def deliver(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def target() -> None:
        try:
            result = func()
        except BaseException as exc:
            outcome: tuple[Callable[[Any], None], Any] = (future.set_exception, exc)
        else:
            outcome = (future.set_result, result)
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:  # loop already closed: nobody is waiting any more
            pass

    threading.Thread(target=target, name="model-loader", daemon=True).start()
    async with asyncio.timeout(timeout_seconds):
        return await future


def _episode(model: Any, seed: int, max_steps: int) -> EpisodeResult:
    def policy(observation: np.ndarray) -> int:
        return int(model.predict(observation.reshape(1, -1))[0])

    # Observations come from the simulator and are finite; skipping sklearn's per-call
    # finiteness scan removes avoidable overhead from hundreds of 1-row predictions.
    with config_context(assume_finite=True):
        return run_episode(policy, seed, max_steps)
