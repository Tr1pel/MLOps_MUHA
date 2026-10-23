import pytest
from pydantic import ValidationError

from agent_platform.core.config import Settings


def test_settings_read_environment_without_exposing_password(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "database.example")
    monkeypatch.setenv("POSTGRES_PASSWORD", "private@:/password")
    settings = Settings(_env_file=None)
    assert settings.database_url.host == "database.example"
    assert settings.database_url.password == "private@:/password"
    assert settings.database_url.drivername == "postgresql+asyncpg"
    assert "private@:/password" not in repr(settings)
    assert "private@:/password" not in str(settings.database_url)


@pytest.mark.parametrize(
    "values",
    [
        {"health_timeout_seconds": 0},
        {"health_timeout_seconds": 31},
        {"postgres_port": 0},
        {"postgres_port": 65536},
        {"log_level": "INVALID"},
    ],
)
def test_invalid_configuration_fails_early(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)
