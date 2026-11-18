from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "agent_platform"
    postgres_user: str = "agent"
    postgres_password: SecretStr = SecretStr("")
    health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_commit: str = "unknown"
    # None disables model loading entirely: the app then runs without MLflow.
    mlflow_tracking_uri: str | None = None
    model_name: str = Field(default="fly-policy", min_length=1)
    model_alias: str = Field(default="champion", min_length=1)
    model_load_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    # Episodes in flight at once; also the size of the dedicated inference thread pool.
    inference_concurrency: int = Field(default=2, ge=1, le=16)
    # How long a request may wait for a free inference slot before getting 503 "busy".
    inference_queue_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    @field_validator("mlflow_tracking_uri", mode="before")
    @classmethod
    def _blank_uri_means_unset(cls, value: object) -> object:
        # An empty MLFLOW_TRACKING_URI= line in .env or the environment means "not configured".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def database_url(self) -> URL:
        # URL.create safely handles passwords containing @, /, : and other URL characters.
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
