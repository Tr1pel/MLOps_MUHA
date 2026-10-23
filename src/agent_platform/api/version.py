from fastapi import APIRouter, Request
from pydantic import BaseModel

from agent_platform.core.version import application_version

router = APIRouter(prefix="/api/v1", tags=["operations"])


class VersionResponse(BaseModel):
    version: str
    commit: str


@router.get("/version", response_model=VersionResponse)
async def get_version(request: Request) -> VersionResponse:
    return VersionResponse(
        version=application_version(), commit=request.app.state.settings.app_commit
    )
