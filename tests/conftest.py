import io
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest
from asgi_lifespan import LifespanManager

from agent_platform.core.config import Settings
from agent_platform.core.logging import JsonFormatter, logger
from agent_platform.main import create_app
from agent_platform.ml.inference import LoadedModel, ModelInfo


@pytest.fixture
def settings():
    return Settings(_env_file=None, app_commit="abc123", health_timeout_seconds=0.1)


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app):
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
def log_output(client, monkeypatch):
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    monkeypatch.setattr(logger, "handlers", [handler])
    return output


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


@pytest.fixture
def model_settings():
    return Settings(
        _env_file=None,
        app_commit="abc123",
        health_timeout_seconds=0.1,
        mlflow_tracking_uri="http://mlflow.test:5000",
        inference_queue_timeout_seconds=0.2,
    )


@pytest.fixture
def started_app():
    """Run an app with an injected model loader inside its lifespan; yields (app, client)."""

    @asynccontextmanager
    async def start(settings, loader):
        app = create_app(settings, model_loader=loader)
        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield app, client

    return start
