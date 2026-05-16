from __future__ import annotations

from uuid import UUID

from app.db.repositories import get_prediction_by_id, list_predictions


def test_batch_predictions_are_persisted_and_total_time_is_returned(
    client_with_fake_classifier,
    db_session,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/batch-analyze",
        json={
            "texts": [
                "Спасибо за внимательное отношение",
                "Когда обновится статус заказа?",
                "Приложение работает нестабильно после релиза",
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    items = payload["items"]

    assert len(items) == 3
    assert payload["processing_time_ms"] >= 0

    persisted_ids = {UUID(item["id"]) for item in items}
    page = list_predictions(db_session, limit=10, offset=0)

    assert page.total == 3
    assert {record.id for record in page.items} == persisted_ids

    for item in items:
        prediction_id = UUID(item["id"])
        record = get_prediction_by_id(db_session, prediction_id)

        assert record is not None
        assert record.id == prediction_id
        assert record.text == item["text"]
        assert record.label == item["label"]
        assert record.confidence == item["confidence"]
        assert record.probabilities == item["probabilities"]
        assert record.processing_time_ms == item["processing_time_ms"]
        assert record.model_name == "local-tfidf-logreg"
        assert record.model_version == "0.1.0"
