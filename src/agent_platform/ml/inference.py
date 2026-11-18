"""Load the registered policy once at startup and run episodes with it.

Concurrency model: an episode is synchronous CPU-bound Python, so it runs in a dedicated
``ThreadPoolExecutor`` of ``inference_concurrency`` workers, never on the event loop. An
``asyncio.Semaphore`` of the same size gates access, so the pool queue is always empty and
excess requests wait (cancellably, with a timeout) on the event loop instead of piling up
in the pool. The semaphore slot is released when the *thread* finishes, not when the
request is cancelled, so the bound holds even if clients disconnect mid-episode. A fitted
scikit-learn pipeline is only read by ``predict`` (no state is mutated), so one shared
instance is used by all workers.
"""

import asyncio
import os
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from sklearn import config_context

from agent_platform.core.config import Settings
from agent_platform.core.logging import logger
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


class ModelBusyError(Exception):
    """No inference slot became free within the configured queue timeout."""


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


class ModelService:
    """Holds the single loaded model (or the reason there is none) for the process lifetime."""

    def __init__(self, settings: Settings, loaded: LoadedModel | None, failure: str | None):
        self.name = settings.model_name
        self.alias = settings.model_alias
        self.loaded = loaded
        self.failure = failure
        self._queue_timeout = settings.inference_queue_timeout_seconds
        self._slots = asyncio.Semaphore(settings.inference_concurrency)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=settings.inference_concurrency, thread_name_prefix="inference"
            )
            if loaded
            else None
        )

    @classmethod
    async def start(
        cls, settings: Settings, loader: ModelLoader = load_registered_model
    ) -> "ModelService":
        """Load once. Never raises: a failed load yields a service that reports why."""
        if settings.mlflow_tracking_uri is None:
            logger.info("model_load_skipped", extra={"reason": NOT_CONFIGURED})
            return cls(settings, None, NOT_CONFIGURED)
        started = perf_counter()
        try:
            loaded = await _run_in_daemon_thread(
                lambda: loader(settings), settings.model_load_timeout_seconds
            )
        except Exception as exc:
            # CancelledError is a BaseException and is deliberately not swallowed here.
            if isinstance(exc, TimeoutError):
                reason = LOAD_TIMEOUT
            elif isinstance(exc, ModelLoadError):
                reason = exc.reason
            else:
                reason = LOAD_FAILED
            logger.error(
                "model_load_failed",
                extra={
                    "reason": reason,
                    "error_type": type(exc).__name__,
                    "model_name": settings.model_name,
                    "model_alias": settings.model_alias,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
            return cls(settings, None, reason)
        info = loaded.info
        logger.info(
            "model_loaded",
            extra={
                "model_name": info.name,
                "model_alias": info.alias,
                "model_version": info.version,
                "run_id": info.run_id,
                "duration_ms": info.load_duration_ms,
            },
        )
        return cls(settings, loaded, None)

    @property
    def info(self) -> ModelInfo | None:
        return self.loaded.info if self.loaded else None

    async def run_episode(self, seed: int, max_steps: int) -> EpisodeResult:
        if self.loaded is None or self._executor is None:
            raise ModelLoadError(self.failure or LOAD_FAILED)
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._slots.acquire()
        except TimeoutError:
            raise ModelBusyError from None
        loop = asyncio.get_running_loop()
        try:
            worker = self._executor.submit(_episode, self.loaded.model, seed, max_steps)
        except BaseException:
            self._slots.release()
            raise

        def release_slot(_: object) -> None:
            try:
                loop.call_soon_threadsafe(self._slots.release)
            except RuntimeError:  # loop closed during shutdown
                pass

        # Free the slot when the worker thread is really done, even if this request is
        # cancelled first (a running thread cannot be cancelled); otherwise abandoned
        # episodes would exceed the concurrency bound.
        worker.add_done_callback(release_slot)
        return await asyncio.wrap_future(worker)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


def _episode(model: Any, seed: int, max_steps: int) -> EpisodeResult:
    def policy(observation: np.ndarray) -> int:
        return int(model.predict(observation.reshape(1, -1))[0])

    # Observations come from the simulator and are finite; skipping sklearn's per-call
    # finiteness scan removes avoidable overhead from hundreds of 1-row predictions.
    with config_context(assume_finite=True):
        return run_episode(policy, seed, max_steps)
