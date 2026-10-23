import asyncio
import tomllib
from importlib.metadata import version
from pathlib import Path


async def test_version_matches_installed_package_and_project(client):
    metadata = await asyncio.to_thread(Path("pyproject.toml").read_text, encoding="utf-8")
    project_version = tomllib.loads(metadata)["project"]["version"]
    response = await client.get("/api/v1/version")
    assert response.status_code == 200
    assert response.json() == {"version": version("agent-platform"), "commit": "abc123"}
    assert response.json()["version"] == project_version
    assert response.json()["version"] != "v1"


async def test_openapi_documents_health_failure_and_application_version(client):
    response = await client.get("/openapi.json")
    schema = response.json()
    assert schema["info"]["version"] == version("agent-platform")
    assert "503" in schema["paths"]["/api/v1/health"]["get"]["responses"]
