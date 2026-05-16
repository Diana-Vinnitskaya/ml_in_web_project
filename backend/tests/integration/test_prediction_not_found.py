from __future__ import annotations

from uuid import uuid4


def test_missing_prediction_id_returns_json_404_without_traceback(
    client_with_fake_classifier,
) -> None:
    response = client_with_fake_classifier.get(f"/api/v1/predictions/{uuid4()}")

    assert response.status_code == 404
    payload = response.json()

    assert payload["detail"] == "Prediction not found"
    assert "request_id" in payload
    assert "Traceback" not in str(payload)
