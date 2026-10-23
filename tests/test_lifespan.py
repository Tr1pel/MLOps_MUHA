from unittest.mock import AsyncMock, MagicMock

import pytest
from asgi_lifespan import LifespanManager

import agent_platform.main as main_module


@pytest.mark.parametrize("raise_inside", [False, True])
async def test_engine_is_lazy_and_disposed_on_shutdown(settings, monkeypatch, raise_inside):
    engine = MagicMock()
    engine.dispose = AsyncMock()
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(main_module, "create_engine", factory)
    app = main_module.create_app(settings)
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
