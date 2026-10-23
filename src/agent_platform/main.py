from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_platform.api.health import router as health_router
from agent_platform.api.version import router as version_router
from agent_platform.core.config import Settings
from agent_platform.core.logging import RequestLoggingMiddleware, configure_logging, logger
from agent_platform.core.version import application_version
from agent_platform.db.connection import create_engine


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        app.state.engine = create_engine(settings)
        logger.info(
            "application_started",
            extra={"version": application_version(), "commit": settings.app_commit},
        )
        try:
            yield
        finally:
            await app.state.engine.dispose()
            logger.info("application_stopped")

    app = FastAPI(
        title="Virtual Fly Agent Platform",
        version=application_version(),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health_router)
    app.include_router(version_router)
    return app


app = create_app()
