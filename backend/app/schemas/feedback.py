from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import get_settings


settings = get_settings()

MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = settings.max_text_length
MAX_BATCH_SIZE = settings.max_batch_size
MAX_HISTORY_LIMIT = 100


class FeedbackLabel(StrEnum):
    complaint = "complaint"
    question = "question"
    praise = "praise"
    other = "other"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeRequest(StrictSchema):
    # Валидация данных
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="Russian feedback text to classify.",
        examples=["Доставка опоздала, я недоволен"],
    )

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class BatchAnalyzeRequest(StrictSchema):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Batch of Russian feedback texts to classify.",
    )

    @field_validator("texts", mode="before")
    @classmethod
    def strip_items(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value

    @field_validator("texts")
    @classmethod
    def validate_items(cls, value: list[str]) -> list[str]:
        for item in value:
            if len(item) < MIN_TEXT_LENGTH:
                raise ValueError(
                    f"Each text must contain at least {MIN_TEXT_LENGTH} characters.",
                )
            if len(item) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Each text must contain at most {MAX_TEXT_LENGTH} characters.",
                )
        return value


class ProbabilityDistribution(StrictSchema):
    complaint: float = Field(..., ge=0, le=1)
    question: float = Field(..., ge=0, le=1)
    praise: float = Field(..., ge=0, le=1)
    other: float = Field(..., ge=0, le=1)


class PredictionResponse(StrictSchema):
    id: UUID
    text: str
    label: FeedbackLabel
    confidence: float = Field(..., ge=0, le=1)
    probabilities: ProbabilityDistribution
    processing_time_ms: float = Field(..., ge=0)
    created_at: datetime


class PredictionDetailResponse(PredictionResponse):
    model_name: str
    model_version: str


class BatchPredictionResponse(StrictSchema):
    items: list[PredictionResponse]
    processing_time_ms: float = Field(..., ge=0)


class PredictionSummary(StrictSchema):
    id: UUID
    text: str
    label: FeedbackLabel
    confidence: float = Field(..., ge=0, le=1)
    created_at: datetime


class PredictionListResponse(StrictSchema):
    items: list[PredictionSummary]
    limit: int = Field(..., ge=1, le=MAX_HISTORY_LIMIT)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    accuracy: float | None = Field(default=None, ge=0, le=1)
    macro_f1: float | None = Field(default=None, ge=0, le=1)


class ModelInfoResponse(StrictSchema):
    model_name: str
    version: str
    labels: list[FeedbackLabel] = Field(min_length=4, max_length=4)
    max_text_length: int = Field(..., ge=MIN_TEXT_LENGTH)
    max_batch_size: int = Field(..., ge=1)
    loaded: bool
    metrics: ModelMetrics | None = None


class LivenessResponse(StrictSchema):
    status: Literal["alive"]
    service: Literal["backend"]


class ReadinessResponse(StrictSchema):
    status: Literal["ready", "unavailable"]
    model_loaded: bool
    database_available: bool
    migrations_applied: bool
    detail: str | None = None


class HealthResponse(StrictSchema):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    database_available: bool
    model_name: str | None = None
    detail: str | None = None


class CountByLabel(StrictSchema):
    complaint: int = Field(..., ge=0)
    question: int = Field(..., ge=0)
    praise: int = Field(..., ge=0)
    other: int = Field(..., ge=0)


class StatsResponse(StrictSchema):
    total_predictions: int = Field(..., ge=0)
    count_by_label: CountByLabel
    average_confidence: float | None = Field(default=None, ge=0, le=1)
    average_processing_time_ms: float | None = Field(default=None, ge=0)
    last_prediction_at: datetime | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    detail: Any
    request_id: str | None = None
