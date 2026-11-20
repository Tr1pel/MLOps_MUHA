import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from faststream.rabbit import RabbitBroker

from agent_platform.api.health import router as health_router
from agent_platform.api.tasks import router as tasks_router
from agent_platform.api.version import router as version_router
from agent_platform.core.config import Settings
from agent_platform.core.logging import RequestLoggingMiddleware, configure_logging, logger
from agent_platform.core.telemetry import instrument_fastapi, setup_telemetry, shutdown_telemetry
from agent_platform.core.version import application_version
from agent_platform.db.connection import create_engine
from agent_platform.db.tasks import ensure_schema
from agent_platform.messaging import declare_topology, make_broker

SERVICE_NAME = "api"
BrokerFactory = Callable[[Settings], RabbitBroker]


def create_app(settings: Settings | None = None, broker_factory: BrokerFactory = make_broker):
    settings = settings or Settings()
    # At import time, not in the lifespan: uvicorn logs "Started server process" before it.
    configure_logging(settings.log_level, SERVICE_NAME)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_telemetry(SERVICE_NAME, settings)
        app.state.engine = create_engine(settings)
        app.state.broker = broker_factory(settings)
        try:
            await ensure_schema(app.state.engine)
            await app.state.broker.start()
            await declare_topology(app.state.broker, settings)
            logger.info(
                "application_started",
                extra={"version": application_version(), "commit": settings.app_commit},
            )
            yield
        finally:
            await app.state.broker.stop()
            await app.state.engine.dispose()
            logger.info("application_stopped")
            await asyncio.to_thread(shutdown_telemetry)

    app = FastAPI(
        title="Virtual Fly Agent Platform",
        version=application_version(),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestLoggingMiddleware)
    instrument_fastapi(app)
    app.include_router(health_router)
    app.include_router(version_router)
    app.include_router(tasks_router)
    return app


app = create_app()
