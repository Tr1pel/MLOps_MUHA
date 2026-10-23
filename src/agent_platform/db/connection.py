from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from agent_platform.core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=settings.health_timeout_seconds,
        connect_args={"timeout": settings.health_timeout_seconds},
        hide_parameters=True,
    )


async def postgres_version(engine: AsyncEngine) -> str:
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT current_setting('server_version')"))
        return result.scalar_one()
