from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import PredictionRecord


DEFAULT_LABELS = get_settings().labels


@dataclass(slots=True)
class PredictionCreatePayload:
    text: str
    label: str
    confidence: float
    probabilities: Mapping[str, float]
    processing_time_ms: float
    model_name: str
    model_version: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime | None = None


@dataclass(slots=True)
class PredictionPage:
    items: list[PredictionRecord]
    total: int


@dataclass(slots=True)
class PredictionStats:
    total_predictions: int
    count_by_label: dict[str, int]
    average_confidence: float | None
    average_processing_time_ms: float | None
    last_prediction_at: datetime | None


def _coerce_payload(
    payload: PredictionCreatePayload | Mapping[str, Any],
) -> PredictionCreatePayload:
    if isinstance(payload, PredictionCreatePayload):
        return payload
    return PredictionCreatePayload(**payload)


def create_prediction(
    session: Session,
    payload: PredictionCreatePayload | Mapping[str, Any],
) -> PredictionRecord:
    record_payload = _coerce_payload(payload)
    record_kwargs: dict[str, Any] = {
        "id": record_payload.id,
        "text": record_payload.text,
        "label": str(record_payload.label),
        "confidence": record_payload.confidence,
        "probabilities": dict(record_payload.probabilities),
        "processing_time_ms": record_payload.processing_time_ms,
        "model_name": record_payload.model_name,
        "model_version": record_payload.model_version,
    }
    if record_payload.created_at is not None:
        record_kwargs["created_at"] = record_payload.created_at

    record = PredictionRecord(**record_kwargs)
    session.add(record)
    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    session.refresh(record)
    return record


def update_prediction_processing_time(
    session: Session,
    record: PredictionRecord,
    processing_time_ms: float,
) -> PredictionRecord:
    record.processing_time_ms = float(processing_time_ms)
    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    session.refresh(record)
    return record


def update_predictions_processing_time(
    session: Session,
    records: Iterable[PredictionRecord],
    processing_time_ms: float,
) -> list[PredictionRecord]:
    updated_records = list(records)
    if not updated_records:
        return []

    for record in updated_records:
        record.processing_time_ms = float(processing_time_ms)

    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise

    for record in updated_records:
        session.refresh(record)
    return updated_records


def create_predictions_batch(
    session: Session,
    payloads: Iterable[PredictionCreatePayload | Mapping[str, Any]],
) -> list[PredictionRecord]:
    records = [
        PredictionRecord(
            **{
                "id": payload.id,
                "text": payload.text,
                "label": str(payload.label),
                "confidence": payload.confidence,
                "probabilities": dict(payload.probabilities),
                "processing_time_ms": payload.processing_time_ms,
                "model_name": payload.model_name,
                "model_version": payload.model_version,
                **(
                    {"created_at": payload.created_at}
                    if payload.created_at is not None
                    else {}
                ),
            },
        )
        for payload in map(_coerce_payload, payloads)
    ]
    if not records:
        return []

    session.add_all(records)
    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
    for record in records:
        session.refresh(record)
    return records


def get_prediction_by_id(
    session: Session,
    prediction_id: UUID | str,
) -> PredictionRecord | None:
    if isinstance(prediction_id, str):
        try:
            prediction_id = UUID(prediction_id)
        except ValueError:
            return None

    return session.get(PredictionRecord, prediction_id)


def list_predictions(
    session: Session,
    *,
    limit: int,
    offset: int = 0,
    label: str | None = None,
) -> PredictionPage:
    filters = []
    if label is not None:
        filters.append(PredictionRecord.label == str(label))

    items_query = (
        select(PredictionRecord)
        .where(*filters)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    total_query = select(func.count()).select_from(PredictionRecord).where(*filters)

    items = list(session.scalars(items_query))
    total = session.scalar(total_query) or 0
    return PredictionPage(items=items, total=total)


def get_prediction_stats(session: Session) -> PredictionStats:
    aggregate_query = select(
        func.count(PredictionRecord.id),
        func.avg(PredictionRecord.confidence),
        func.avg(PredictionRecord.processing_time_ms),
        func.max(PredictionRecord.created_at),
    )
    total, average_confidence, average_processing_time_ms, last_prediction_at = (
        session.execute(aggregate_query).one()
    )

    grouped_counts = dict(
        session.execute(
            select(PredictionRecord.label, func.count(PredictionRecord.id)).group_by(
                PredictionRecord.label,
            ),
        ).all(),
    )
    count_by_label = {label: int(grouped_counts.get(label, 0)) for label in DEFAULT_LABELS}

    return PredictionStats(
        total_predictions=int(total or 0),
        count_by_label=count_by_label,
        average_confidence=float(average_confidence) if average_confidence is not None else None,
        average_processing_time_ms=(
            float(average_processing_time_ms)
            if average_processing_time_ms is not None
            else None
        ),
        last_prediction_at=last_prediction_at,
    )
