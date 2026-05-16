from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.db.repositories import PredictionCreatePayload, create_prediction
from app.schemas.feedback import (
    PredictionDetailResponse,
    PredictionListResponse,
    StatsResponse,
)


def make_probabilities(label: str, confidence: float) -> dict[str, float]:
    probabilities = {
        "complaint": 0.03,
        "question": 0.03,
        "praise": 0.03,
        "other": 0.03,
    }
    probabilities[label] = confidence
    return probabilities


def seed_prediction(
    db_session,
    *,
    text: str,
    label: str,
    confidence: float,
    created_at: datetime,
):
    return create_prediction(
        db_session,
        PredictionCreatePayload(
            text=text,
            label=label,
            confidence=confidence,
            probabilities=make_probabilities(label, confidence),
            processing_time_ms=12.5,
            model_name="local-tfidf-logreg",
            model_version="0.1.0",
            created_at=created_at,
        ),
    )


def test_predictions_list_returns_bounded_history_with_total(
    client_with_fake_classifier,
    db_session,
) -> None:
    now = datetime.now(timezone.utc)
    older = seed_prediction(
        db_session,
        text="Старый отзыв с проблемой доставки",
        label="complaint",
        confidence=0.88,
        created_at=now - timedelta(minutes=5),
    )
    newer = seed_prediction(
        db_session,
        text="Новый отзыв с благодарностью",
        label="praise",
        confidence=0.91,
        created_at=now,
    )

    response = client_with_fake_classifier.get(
        "/api/v1/predictions",
        params={"limit": 1, "offset": 0},
    )

    parsed = PredictionListResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.limit == 1
    assert parsed.offset == 0
    assert parsed.total == 2
    assert len(parsed.items) == 1
    assert parsed.items[0].id == newer.id
    assert parsed.items[0].label == "praise"
    assert parsed.items[0].confidence == newer.confidence
    assert older.id != parsed.items[0].id


def test_predictions_list_supports_label_filter(
    client_with_fake_classifier,
    db_session,
) -> None:
    now = datetime.now(timezone.utc)
    seed_prediction(
        db_session,
        text="Почему заказ задерживается?",
        label="question",
        confidence=0.84,
        created_at=now - timedelta(minutes=1),
    )
    expected = seed_prediction(
        db_session,
        text="Спасибо за подробный ответ",
        label="praise",
        confidence=0.93,
        created_at=now,
    )

    response = client_with_fake_classifier.get(
        "/api/v1/predictions",
        params={"label": "praise", "limit": 20, "offset": 0},
    )

    parsed = PredictionListResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.total == 1
    assert len(parsed.items) == 1
    assert parsed.items[0].id == expected.id
    assert parsed.items[0].label == "praise"


def test_predictions_list_returns_422_for_invalid_label(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get(
        "/api/v1/predictions",
        params={"label": "invalid-label"},
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert "request_id" in response.json()


def test_prediction_detail_returns_full_response_shape(
    client_with_fake_classifier,
    db_session,
) -> None:
    record = seed_prediction(
        db_session,
        text="Спасибо за быстрое решение проблемы",
        label="praise",
        confidence=0.95,
        created_at=datetime.now(timezone.utc),
    )

    response = client_with_fake_classifier.get(f"/api/v1/predictions/{record.id}")

    parsed = PredictionDetailResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.id == record.id
    assert parsed.text == record.text
    assert parsed.label == "praise"
    assert parsed.confidence == record.confidence
    assert parsed.probabilities.praise == 0.95
    assert parsed.model_name == "local-tfidf-logreg"
    assert parsed.model_version == "0.1.0"


def test_prediction_detail_returns_404_for_missing_id(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get(f"/api/v1/predictions/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"
    assert "request_id" in response.json()


def test_stats_return_aggregate_schema_and_counts(
    client_with_fake_classifier,
    db_session,
) -> None:
    now = datetime.now(timezone.utc)
    seed_prediction(
        db_session,
        text="Опоздала доставка, я недоволен",
        label="complaint",
        confidence=0.82,
        created_at=now - timedelta(minutes=2),
    )
    seed_prediction(
        db_session,
        text="Спасибо за оперативную поддержку",
        label="praise",
        confidence=0.94,
        created_at=now,
    )

    response = client_with_fake_classifier.get("/api/v1/stats")

    parsed = StatsResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.total_predictions == 2
    assert parsed.count_by_label.complaint == 1
    assert parsed.count_by_label.praise == 1
    assert parsed.count_by_label.question == 0
    assert parsed.count_by_label.other == 0
    assert parsed.average_confidence is not None
    assert parsed.average_processing_time_ms is not None
    assert parsed.last_prediction_at is not None


def test_stats_return_503_when_database_dependency_is_unavailable(
    client_with_fake_classifier,
) -> None:
    client_with_fake_classifier.app.state.session_factory = None

    response = client_with_fake_classifier.get("/api/v1/stats")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database session factory is unavailable"
    assert "request_id" in response.json()
