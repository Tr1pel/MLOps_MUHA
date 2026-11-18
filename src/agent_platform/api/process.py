from typing import Annotated, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_platform.core.logging import logger
from agent_platform.ml.inference import ModelBusyError, ModelInfo, ModelLoadError
from agent_platform.sim.env import DEFAULT_MAX_STEPS

router = APIRouter(tags=["inference"])

MAX_STEPS_LIMIT = 1000


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: Annotated[int, Field(ge=0, le=2**32 - 1, description="Episode (world) seed")]
    max_steps: Annotated[int, Field(ge=1, le=MAX_STEPS_LIMIT)] = DEFAULT_MAX_STEPS


class ModelRef(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str | None


class Obstacle(BaseModel):
    x: float
    y: float
    radius: float


class ProcessResponse(BaseModel):
    seed: int
    success: bool
    steps: int
    collisions: int
    trajectory: list[tuple[float, float]]
    food: tuple[float, float]
    obstacles: list[Obstacle]
    model: ModelRef


class ErrorResponse(BaseModel):
    error: Literal["model_unavailable", "busy", "inference_failed"]
    reason: str | None = Field(
        default=None,
        description="For model_unavailable: not_configured, alias_not_found, timeout or load_failed.",
    )


def model_ref(info: ModelInfo) -> ModelRef:
    return ModelRef(name=info.name, alias=info.alias, version=info.version, run_id=info.run_id)


def _error(status: int, body: ErrorResponse, headers: dict[str, str] | None = None):
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


@router.post(
    "/process",
    response_model=ProcessResponse,
    responses={
        503: {
            "model": ErrorResponse,
            "description": "No model loaded at startup (see reason) or all inference slots busy",
        },
        500: {"model": ErrorResponse, "description": "Unexpected inference failure"},
    },
)
async def process(payload: ProcessRequest, request: Request):
    """Run one episode of the virtual fly with the model loaded at startup.

    The model is never loaded here: a restart is what picks up a moved registry alias.
    """
    service = request.app.state.model_service
    try:
        result = await service.run_episode(payload.seed, payload.max_steps)
    except ModelLoadError as exc:
        return _error(503, ErrorResponse(error="model_unavailable", reason=exc.reason))
    except ModelBusyError:
        return _error(503, ErrorResponse(error="busy"), headers={"Retry-After": "1"})
    except Exception as exc:
        # Stable body for clients; the exception message stays out of logs and responses.
        logger.error(
            "inference_failed",
            extra={"error_type": type(exc).__name__, "model_version": service.info.version},
        )
        return _error(500, ErrorResponse(error="inference_failed"))
    return ProcessResponse(
        seed=result.seed,
        success=result.success,
        steps=result.steps,
        collisions=result.collisions,
        trajectory=result.trajectory,
        food=result.food,
        obstacles=[Obstacle(x=x, y=y, radius=r) for x, y, r in result.obstacles],
        model=model_ref(service.info),
    )
