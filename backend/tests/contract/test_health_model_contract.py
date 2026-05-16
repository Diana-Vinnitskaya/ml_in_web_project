from __future__ import annotations

from app.schemas.feedback import (
    HealthResponse,
    LivenessResponse,
    ModelInfoResponse,
    ReadinessResponse,
)


def test_liveness_returns_backend_alive(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get("/api/v1/health/live")

    parsed = LivenessResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.status == "alive"
    assert parsed.service == "backend"


def test_readiness_returns_ready_response_shape(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get("/api/v1/health/ready")

    parsed = ReadinessResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.status == "ready"
    assert parsed.model_loaded is True
    assert parsed.database_available is True
    assert parsed.migrations_applied is True
    assert parsed.detail is None


def test_health_returns_ok_response_with_model_name(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get("/api/v1/health")

    parsed = HealthResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.status == "ok"
    assert parsed.model_loaded is True
    assert parsed.database_available is True
    assert parsed.model_name == "local-tfidf-logreg"
    assert parsed.detail is None


def test_model_info_returns_loaded_profile(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get("/api/v1/model/info")

    parsed = ModelInfoResponse.model_validate(response.json())

    assert response.status_code == 200
    assert parsed.model_name == "local-tfidf-logreg"
    assert parsed.version == "0.1.0"
    assert [label.value for label in parsed.labels] == [
        "complaint",
        "question",
        "praise",
        "other",
    ]
    assert parsed.max_text_length == 2000
    assert parsed.max_batch_size == 32
    assert parsed.loaded is True
    assert parsed.metrics is not None
    assert parsed.metrics.accuracy == 0.95
    assert parsed.metrics.macro_f1 == 0.95


def test_readiness_returns_503_when_model_is_unavailable(
    client_with_unavailable_classifier,
) -> None:
    response = client_with_unavailable_classifier.get("/api/v1/health/ready")

    parsed = ReadinessResponse.model_validate(response.json())

    assert response.status_code == 503
    assert parsed.status == "unavailable"
    assert parsed.model_loaded is False
    assert parsed.database_available is True
    assert parsed.migrations_applied is True
    assert parsed.detail == "Model is not loaded"


def test_model_info_returns_503_when_model_is_unavailable(
    client_with_unavailable_classifier,
) -> None:
    response = client_with_unavailable_classifier.get("/api/v1/model/info")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"
    assert "request_id" in response.json()
