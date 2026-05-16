from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, Index, String, Text, desc, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator, Uuid

from app.db.base import Base


PROBABILITIES_TYPE = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator[datetime]):
    """Preserve UTC-aware datetimes across Postgres and SQLite test runs."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        _dialect: Any,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(
        self,
        value: datetime | None,
        _dialect: Any,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PredictionRecord(Base):
    """Durable prediction history row."""

    __tablename__ = "prediction_records"
    __table_args__ = (
        Index("ix_prediction_records_created_at_desc", desc("created_at")),
        Index("ix_prediction_records_label", "label"),
    )

    # ORM
    id: Mapped[UUID] = mapped_column(
        Uuid(),
        primary_key=True,
        default=uuid4,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict[str, float]] = mapped_column(
        PROBABILITIES_TYPE,
        nullable=False,
    )
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )


class ModelTrainingRun(Base):
    """Optional durable training evidence for the current model artifact."""

    __tablename__ = "model_training_runs"
    __table_args__ = (
        Index("ix_model_training_runs_trained_at_desc", desc("trained_at")),
        Index(
            "ix_model_training_runs_model_identity",
            "model_name",
            "model_version",
        ),
    )

    # ORM
    id: Mapped[UUID] = mapped_column(
        Uuid(),
        primary_key=True,
        default=uuid4,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    macro_f1: Mapped[float] = mapped_column(Float, nullable=False)
    classification_report: Mapped[dict[str, Any] | None] = mapped_column(
        PROBABILITIES_TYPE,
        nullable=True,
    )
    train_size: Mapped[int] = mapped_column(nullable=False)
    test_size: Mapped[int] = mapped_column(nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
