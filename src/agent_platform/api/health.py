import asyncio
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from agent_platform.core.logging import logger
from agent_platform.db.connection import postgres_version

router = APIRouter(tags=["operations"])


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ComponentHealth(BaseModel):
    status: Literal["ok", "error"]
    version: str | None
    response_time_ms: float = Field(ge=0)
    error: Literal["timeout", "unavailable"] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    components: dict[str, ComponentHealth]


@router.get("/healthz", response_model=LivenessResponse)
async def healthz() -> LivenessResponse:
    """Liveness is independent of databases and API versioning."""
    return LivenessResponse()


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Dependency unavailable"}},
)
async def health(request: Request, response: Response) -> HealthResponse:
    started = perf_counter()
    version = None
    error = None
    try:
        # Includes pool acquisition, connection, query and connection release.
        async with asyncio.timeout(request.app.state.settings.health_timeout_seconds):
            version = await postgres_version(request.app.state.engine)
    except Exception as exc:
        # A health boundary converts driver failures to a stable, secret-free contract.
        # CancelledError inherits BaseException, so shutdown cancellation is not swallowed.
        error = "timeout" if isinstance(exc, TimeoutError) else "unavailable"
        logger.warning(
            "dependency_check_failed",
            extra={
                "component": "postgresql",
                "error_type": type(exc).__name__,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            },
        )

    component = ComponentHealth(
        status="error" if error else "ok",
        version=version,
        response_time_ms=round((perf_counter() - started) * 1000, 3),
        error=error,
    )
    if error:
        response.status_code = 503
    return HealthResponse(status=component.status, components={"postgresql": component})
