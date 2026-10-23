import os
import re

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agent_platform.core.config import Settings
from agent_platform.db.connection import postgres_version
from agent_platform.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1"),
]


async def test_health_against_real_postgresql():
    app = create_app(Settings())
    async with LifespanManager(app):
        actual_version = await postgres_version(app.state.engine)
        assert re.match(r"\d+\.\d+", actual_version)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == 200
        component = response.json()["components"]["postgresql"]
        assert component["version"] == actual_version
        assert component["response_time_ms"] >= 0
