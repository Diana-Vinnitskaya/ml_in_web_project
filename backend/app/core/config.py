from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    project_name: str = Field(
        default="RuFeedback Classifier",
        validation_alias="PROJECT_NAME",
    )
    api_prefix: str = Field(default="/api/v1", validation_alias="API_PREFIX")
    model_name: str = Field(
        default="local-tfidf-logreg",
        validation_alias="MODEL_NAME",
    )
    model_version: str = Field(default="0.1.0", validation_alias="MODEL_VERSION")

    max_text_length: int = Field(
        default=2000,
        ge=3,
        validation_alias="MAX_TEXT_LENGTH",
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        validation_alias="MAX_BATCH_SIZE",
    )

    model_path: Path = Field(
        default=Path("/app/models/feedback_classifier.joblib"),
        validation_alias="MODEL_PATH",
    )
    model_metrics_path: Path = Field(
        default=Path("/app/models/metrics.json"),
        validation_alias="MODEL_METRICS_PATH",
    )
    train_data_path: Path = Field(
        default=Path("/app/data/train.csv"),
        validation_alias="TRAIN_DATA_PATH",
    )

    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    postgres_db: str = Field(default="rufeedback", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(
        default="rufeedback",
        validation_alias="POSTGRES_USER",
    )
    postgres_password: str = Field(
        default="rufeedback",
        validation_alias="POSTGRES_PASSWORD",
    )
    postgres_host: str = Field(default="postgres", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")

    backend_host: str = Field(default="0.0.0.0", validation_alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, validation_alias="BACKEND_PORT")
    ui_host: str = Field(default="0.0.0.0", validation_alias="UI_HOST")
    ui_port: int = Field(default=8501, validation_alias="UI_PORT")
    nginx_port: int = Field(default=80, validation_alias="NGINX_PORT")

    prometheus_scrape_interval: str = Field(
        default="15s",
        validation_alias="PROMETHEUS_SCRAPE_INTERVAL",
    )
    grafana_admin_user: str = Field(
        default="admin",
        validation_alias="GRAFANA_ADMIN_USER",
    )
    grafana_admin_password: str = Field(
        default="admin",
        validation_alias="GRAFANA_ADMIN_PASSWORD",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        cleaned = value.strip() or "/api/v1"
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        if cleaned != "/":
            cleaned = cleaned.rstrip("/")
        return cleaned

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def populate_database_url(self) -> "Settings":
        if not self.database_url:
            self.database_url = (
                "postgresql+psycopg://"
                f"{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @property
    def model_dir(self) -> Path:
        return self.model_path.parent

    @property
    def labels(self) -> tuple[str, str, str, str]:
        return ("complaint", "question", "praise", "other")


@lru_cache
def get_settings() -> Settings:
    return Settings()
