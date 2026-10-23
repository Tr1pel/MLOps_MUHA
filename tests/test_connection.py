from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.db.connection import postgres_version


async def test_probe_executes_server_query_and_releases_connection():
    connection = AsyncMock()
    result = MagicMock()
    result.scalar_one.return_value = "17.6"
    connection.execute.return_value = result
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=connection)
    context.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = context

    assert await postgres_version(engine) == "17.6"
    query = connection.execute.await_args.args[0]
    assert str(query) == "SELECT current_setting('server_version')"
    context.__aexit__.assert_awaited_once()


async def test_probe_releases_connection_on_query_failure():
    connection = AsyncMock()
    connection.execute.side_effect = RuntimeError("query failed")
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=connection)
    context.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = context
    with pytest.raises(RuntimeError, match="query failed"):
        await postgres_version(engine)
    context.__aexit__.assert_awaited_once()
