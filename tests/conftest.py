import io
import logging

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agent_platform.core.config import Settings
from agent_platform.core.logging import JsonFormatter, logger
from agent_platform.main import create_app


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
