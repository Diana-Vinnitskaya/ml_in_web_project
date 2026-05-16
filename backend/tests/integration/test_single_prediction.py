from __future__ import annotations

from uuid import UUID

from app.db.repositories import get_prediction_by_id, list_predictions


def test_single_prediction_is_persisted_and_returned(
    client_with_fake_classifier,
    db_session,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/analyze",
        json={"text": "Спасибо за внимательное отношение к клиенту"},
    )

    assert response.status_code == 200
    payload = response.json()
    prediction_id = UUID(payload["id"])

    record = get_prediction_by_id(db_session, prediction_id)
    page = list_predictions(db_session, limit=10, offset=0)

    assert record is not None
    assert record.id == prediction_id
    assert record.text == payload["text"]
    assert record.label == payload["label"]
    assert record.confidence == payload["confidence"]
    assert record.probabilities == payload["probabilities"]
    assert record.processing_time_ms == payload["processing_time_ms"]
    assert record.model_name == "local-tfidf-logreg"
    assert record.model_version == "0.1.0"
    assert page.total == 1
    assert page.items[0].id == prediction_id
