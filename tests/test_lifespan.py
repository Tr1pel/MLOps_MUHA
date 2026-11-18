from unittest.mock import AsyncMock, MagicMock

import pytest
from asgi_lifespan import LifespanManager
from conftest import FakeBroker

import agent_platform.main as main_module


@pytest.mark.parametrize("raise_inside", [False, True])
async def test_engine_is_lazy_and_disposed_on_shutdown(settings, monkeypatch, raise_inside):
    engine = MagicMock()
    engine.dispose = AsyncMock()
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(main_module, "create_engine", factory)
    app = main_module.create_app(settings, broker_factory=FakeBroker)
    try:
        async with LifespanManager(app):
            assert app.state.engine is engine
            engine.connect.assert_not_called()
            if raise_inside:
                raise RuntimeError("test shutdown on failure")
    except RuntimeError:
        if not raise_inside:
            raise
    factory.assert_called_once_with(settings)
    engine.dispose.assert_awaited_once()


async def test_startup_prepares_schema_and_queue_then_stops_broker(settings, monkeypatch):
    calls = []
    engine = MagicMock()
    engine.dispose = AsyncMock()
    broker = FakeBroker()
    broker.start.side_effect = lambda: calls.append("start")
    broker.stop.side_effect = lambda: calls.append("stop")
    monkeypatch.setattr(main_module, "create_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(
        main_module, "ensure_schema", AsyncMock(side_effect=lambda e: calls.append("schema"))
    )
    monkeypatch.setattr(
        main_module,
        "declare_topology",
        AsyncMock(side_effect=lambda b, s: calls.append("topology")),
    )
    app = main_module.create_app(settings, broker_factory=lambda s: broker)
    async with LifespanManager(app):
        assert app.state.broker is broker
    assert calls == ["schema", "start", "topology", "stop"]
    engine.dispose.assert_awaited_once()


async def test_resources_are_released_when_startup_fails(settings, monkeypatch):
    broker = FakeBroker()
    engine = MagicMock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr(main_module, "create_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(main_module, "ensure_schema", AsyncMock(side_effect=OSError("db down")))
    app = main_module.create_app(settings, broker_factory=lambda s: broker)
    with pytest.raises(OSError):
        async with LifespanManager(app):
            pass
    broker.stop.assert_awaited_once()
    engine.dispose.assert_awaited_once()
