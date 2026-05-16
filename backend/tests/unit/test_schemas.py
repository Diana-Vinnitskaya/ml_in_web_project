from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.feedback import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchPredictionResponse,
    ErrorResponse,
    PredictionResponse,
    ProbabilityDistribution,
)


def test_analyze_request_strips_text_before_validation() -> None:
    payload = AnalyzeRequest(text="   Спасибо за поддержку!   ")

    assert payload.text == "Спасибо за поддержку!"


@pytest.mark.parametrize("text", ["", "  ", "ok", "  a "])
def test_analyze_request_rejects_short_text(text: str) -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(text=text)


def test_analyze_request_rejects_too_long_text() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="а" * 2001)


def test_batch_analyze_request_strips_each_item() -> None:
    payload = BatchAnalyzeRequest(
        texts=[
            "  Спасибо за помощь  ",
            "Когда ответите по заказу?",
        ]
    )

    assert payload.texts == ["Спасибо за помощь", "Когда ответите по заказу?"]


def test_batch_analyze_request_rejects_empty_and_oversized_batches() -> None:
    with pytest.raises(ValidationError):
        BatchAnalyzeRequest(texts=[])

    with pytest.raises(ValidationError):
        BatchAnalyzeRequest(texts=["Текст"] * 33)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probability_distribution_enforces_bounds(value: float) -> None:
    with pytest.raises(ValidationError):
        ProbabilityDistribution(
            complaint=value,
            question=0.25,
            praise=0.25,
            other=0.25,
        )


def test_prediction_response_json_shape_matches_contract() -> None:
    created_at = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    probabilities = ProbabilityDistribution(
        complaint=0.72,
        question=0.1,
        praise=0.08,
        other=0.1,
    )
    response = PredictionResponse(
        id=uuid4(),
        text="Поддержка не отвечает третий день",
        label="complaint",
        confidence=0.72,
        probabilities=probabilities,
        processing_time_ms=123.45,
        created_at=created_at,
    )

    dumped = response.model_dump(mode="json")

    assert set(dumped) == {
        "id",
        "text",
        "label",
        "confidence",
        "probabilities",
        "processing_time_ms",
        "created_at",
    }
    assert dumped["label"] == "complaint"
    assert dumped["probabilities"]["question"] == 0.1
    assert dumped["created_at"] == "2026-05-10T12:00:00Z"


def test_batch_prediction_response_embeds_prediction_items() -> None:
    item = PredictionResponse(
        id=uuid4(),
        text="Спасибо!",
        label="praise",
        confidence=0.91,
        probabilities=ProbabilityDistribution(
            complaint=0.02,
            question=0.02,
            praise=0.91,
            other=0.05,
        ),
        processing_time_ms=45.0,
        created_at=datetime(2026, 5, 10, 12, 30, tzinfo=UTC),
    )
    response = BatchPredictionResponse(items=[item], processing_time_ms=45.0)

    dumped = response.model_dump(mode="json")

    assert len(dumped["items"]) == 1
    assert dumped["items"][0]["label"] == "praise"
    assert dumped["processing_time_ms"] == 45.0


def test_error_response_accepts_structured_detail() -> None:
    response = ErrorResponse(detail=[{"loc": ["body", "text"], "msg": "too short"}])

    assert isinstance(response.detail, list)
