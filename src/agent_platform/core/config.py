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

    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = Field(default=5672, ge=1, le=65535)
    rabbitmq_user: str = "guest"
    rabbitmq_password: SecretStr = SecretStr("guest")
    tasks_queue: str = Field(default="tasks", min_length=1)
    # 0 means an unbounded queue (only for the "no backpressure" experiment).
    queue_max_length: int = Field(default=50, ge=0)
    # Total processing attempts per message before the worker sends it to the dead-letter queue.
    task_delivery_limit: int = Field(default=3, ge=1)
    # Target duration of one task: the worker pads shorter inferences up to this time.
    inference_seconds: float = Field(default=0.5, ge=0)
    # Prefetch count, concurrently handled messages and inference threads, all in one knob.
    worker_concurrency: int = Field(default=2, ge=1, le=64)
    worker_health_port: int = Field(default=8001, ge=1, le=65535)
    publish_timeout_seconds: float = Field(default=2.0, gt=0)

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

    @property
    def rabbitmq_url(self) -> str:
        url = URL.create(
            "amqp",
            username=self.rabbitmq_user,
            password=self.rabbitmq_password.get_secret_value(),
            host=self.rabbitmq_host,
            port=self.rabbitmq_port,
        )
        return url.render_as_string(hide_password=False)
