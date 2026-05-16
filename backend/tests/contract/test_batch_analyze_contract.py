from __future__ import annotations

from uuid import UUID

from app.schemas.feedback import BatchPredictionResponse


def test_batch_analyze_returns_response_shape(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/batch-analyze",
        json={
            "texts": [
                "Спасибо за быструю помощь",
                "Когда приедет мой заказ?",
                "Приложение снова зависает",
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert set(payload) == {"items", "processing_time_ms"}
    assert payload["processing_time_ms"] >= 0
    assert len(payload["items"]) == 3

    for item in payload["items"]:
        assert set(item) == {
            "id",
            "text",
            "label",
            "confidence",
            "probabilities",
            "processing_time_ms",
            "created_at",
        }
        assert item["label"] == "praise"
        assert set(item["probabilities"]) == {
            "complaint",
            "question",
            "praise",
            "other",
        }
        UUID(item["id"])


def test_batch_analyze_response_matches_schema(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/batch-analyze",
        json={
            "texts": [
                "Спасибо, все было понятно",
                "Сколько ждать ответа поддержки?",
            ],
        },
    )

    parsed = BatchPredictionResponse.model_validate(response.json())

    assert response.status_code == 200
    assert len(parsed.items) == 2
    assert parsed.items[0].label == "praise"


def test_batch_analyze_rejects_empty_batch(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/batch-analyze",
        json={"texts": []},
    )

    assert response.status_code == 422
    payload = response.json()

    assert isinstance(payload["detail"], list)
    assert payload["detail"][0]["loc"][-1] == "texts"
    assert "request_id" in payload


def test_batch_analyze_rejects_oversized_batch(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/batch-analyze",
        json={"texts": ["Спасибо за сервис"] * 33},
    )

    assert response.status_code == 422
    payload = response.json()

    assert isinstance(payload["detail"], list)
    assert payload["detail"][0]["loc"][-1] == "texts"


def test_batch_analyze_rejects_invalid_item(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.post(
        "/api/v1/batch-analyze",
        json={"texts": ["Спасибо", "  "]},
    )

    assert response.status_code == 422
    payload = response.json()

    assert isinstance(payload["detail"], list)
    assert payload["detail"][0]["loc"][-1] == "texts"


def test_batch_analyze_returns_503_when_model_is_unavailable(
    client_with_unavailable_classifier,
) -> None:
    response = client_with_unavailable_classifier.post(
        "/api/v1/batch-analyze",
        json={
            "texts": [
                "Где моя доставка?",
                "Поддержка не отвечает уже сутки",
            ],
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"
