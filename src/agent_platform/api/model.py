from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["inference"])


class ModelStatus(BaseModel):
    loaded: bool
    name: str
    alias: str
    version: str | None = None
    run_id: str | None = None
    source: str | None = None
    loaded_at: datetime | None = None
    load_duration_ms: float | None = None
    # Category only (not_configured, alias_not_found, timeout, load_failed), never details.
    reason: str | None = None


@router.get("/model", response_model=ModelStatus)
async def get_model(request: Request) -> ModelStatus:
    """Which registry version is this process serving? Fixed at startup."""
    service = request.app.state.model_service
    info = service.info
    if info is None:
        return ModelStatus(
            loaded=False, name=service.name, alias=service.alias, reason=service.failure
        )
    return ModelStatus(loaded=True, **vars(info))
