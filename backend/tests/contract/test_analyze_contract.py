from __future__ import annotations

from uuid import UUID

from app.schemas.feedback import PredictionResponse


def test_analyze_returns_prediction_response_shape(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/analyze",
        json={"text": "Спасибо за быструю помощь и вежливый ответ"},
    )

    assert response.status_code == 200
    payload = response.json()

    assert set(payload) == {
        "id",
        "text",
        "label",
        "confidence",
        "probabilities",
        "processing_time_ms",
        "created_at",
    }
    assert payload["label"] == "praise"
    assert payload["text"] == "Спасибо за быструю помощь и вежливый ответ"
    assert set(payload["probabilities"]) == {
        "complaint",
        "question",
        "praise",
        "other",
    }
    assert payload["probabilities"]["praise"] == 0.91
    assert payload["processing_time_ms"] >= 0
    UUID(payload["id"])


def test_analyze_response_matches_prediction_schema(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/analyze",
        json={"text": "Поддержка все объяснила очень понятно"},
    )

    parsed = PredictionResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.label == "praise"
    assert parsed.confidence == 0.91


def test_analyze_returns_structured_validation_error(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/analyze",
        json={"text": "  "},
    )

    assert response.status_code == 422
    payload = response.json()

    assert isinstance(payload["detail"], list)
    assert payload["detail"][0]["loc"][-1] == "text"
    assert "request_id" in payload


def test_analyze_returns_503_when_model_is_unavailable(
    client_with_unavailable_classifier,
) -> None:
    response = client_with_unavailable_classifier.post(
        "/api/v1/analyze",
        json={"text": "Где мой заказ, он давно должен был приехать"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"
