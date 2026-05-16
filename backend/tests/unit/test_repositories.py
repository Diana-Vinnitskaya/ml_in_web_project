from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.db.base import Base
from app.db.repositories import (
    PredictionCreatePayload,
    create_prediction,
    create_predictions_batch,
    get_prediction_by_id,
    get_prediction_stats,
    list_predictions,
)
from app.db.session import create_engine_and_session_factory


@pytest.fixture
def session():
    engine, session_factory = create_engine_and_session_factory(
        database_url="sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    active_session = session_factory()
    try:
        yield active_session
    finally:
        active_session.close()
        engine.dispose()


def make_payload(
    *,
    text: str,
    label: str,
    confidence: float,
    processing_time_ms: float,
    created_at: datetime,
) -> PredictionCreatePayload:
    base_probabilities = {
        "complaint": 0.1,
        "question": 0.1,
        "praise": 0.1,
        "other": 0.1,
    }
    base_probabilities[label] = confidence

    return PredictionCreatePayload(
        id=uuid4(),
        text=text,
        label=label,
        confidence=confidence,
        probabilities=base_probabilities,
        processing_time_ms=processing_time_ms,
        model_name="local-tfidf-logreg",
        model_version="0.1.0",
        created_at=created_at,
    )


def test_create_prediction_persists_full_record(session) -> None:
    created_at = datetime(2026, 5, 10, 15, 0, tzinfo=UTC)

    record = create_prediction(
        session,
        make_payload(
            text="Поддержка не отвечает уже неделю",
            label="complaint",
            confidence=0.82,
            processing_time_ms=147.2,
            created_at=created_at,
        ),
    )

    assert record.text == "Поддержка не отвечает уже неделю"
    assert record.label == "complaint"
    assert record.confidence == pytest.approx(0.82)
    assert record.probabilities["complaint"] == pytest.approx(0.82)
    assert record.created_at == created_at


def test_create_predictions_batch_returns_records_in_input_order(session) -> None:
    base_time = datetime(2026, 5, 10, 15, 5, tzinfo=UTC)
    payloads = [
        make_payload(
            text="Когда привезете заказ?",
            label="question",
            confidence=0.77,
            processing_time_ms=91.0,
            created_at=base_time,
        ),
        make_payload(
            text="Спасибо за быструю замену товара",
            label="praise",
            confidence=0.88,
            processing_time_ms=88.0,
            created_at=base_time + timedelta(seconds=1),
        ),
    ]

    records = create_predictions_batch(session, payloads)

    assert [record.id for record in records] == [payload.id for payload in payloads]
    assert [record.label for record in records] == ["question", "praise"]


def test_list_predictions_supports_ordering_filter_and_total(session) -> None:
    base_time = datetime(2026, 5, 10, 15, 10, tzinfo=UTC)
    create_predictions_batch(
        session,
        [
            make_payload(
                text="Почему отменили доставку?",
                label="question",
                confidence=0.75,
                processing_time_ms=110.0,
                created_at=base_time,
            ),
            make_payload(
                text="Все отлично, спасибо!",
                label="praise",
                confidence=0.93,
                processing_time_ms=72.0,
                created_at=base_time + timedelta(minutes=1),
            ),
            make_payload(
                text="Опять не работает оплата",
                label="complaint",
                confidence=0.89,
                processing_time_ms=130.0,
                created_at=base_time + timedelta(minutes=2),
            ),
        ],
    )

    filtered_page = list_predictions(session, limit=10, offset=0, label="praise")
    full_page = list_predictions(session, limit=2, offset=0)

    assert filtered_page.total == 1
    assert [item.label for item in filtered_page.items] == ["praise"]
    assert full_page.total == 3
    assert [item.label for item in full_page.items] == ["complaint", "praise"]


def test_get_prediction_by_id_returns_none_for_unknown_id(session) -> None:
    assert get_prediction_by_id(session, uuid4()) is None
    assert get_prediction_by_id(session, "not-a-uuid") is None


def test_get_prediction_stats_aggregates_counts_and_averages(session) -> None:
    base_time = datetime(2026, 5, 10, 15, 20, tzinfo=UTC)
    create_predictions_batch(
        session,
        [
            make_payload(
                text="Курьер опоздал",
                label="complaint",
                confidence=0.8,
                processing_time_ms=100.0,
                created_at=base_time,
            ),
            make_payload(
                text="Где посмотреть статус заказа?",
                label="question",
                confidence=0.6,
                processing_time_ms=80.0,
                created_at=base_time + timedelta(minutes=1),
            ),
            make_payload(
                text="Спасибо за вежливую поддержку",
                label="praise",
                confidence=0.9,
                processing_time_ms=60.0,
                created_at=base_time + timedelta(minutes=2),
            ),
        ],
    )

    stats = get_prediction_stats(session)

    assert stats.total_predictions == 3
    assert stats.count_by_label == {
        "complaint": 1,
        "question": 1,
        "praise": 1,
        "other": 0,
    }
    assert stats.average_confidence == pytest.approx((0.8 + 0.6 + 0.9) / 3)
    assert stats.average_processing_time_ms == pytest.approx((100.0 + 80.0 + 60.0) / 3)
    assert stats.last_prediction_at == base_time + timedelta(minutes=2)
