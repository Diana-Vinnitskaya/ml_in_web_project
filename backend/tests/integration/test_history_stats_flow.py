from __future__ import annotations

from app.schemas.feedback import PredictionListResponse, StatsResponse


def test_history_and_stats_change_after_new_prediction(
    client_with_fake_classifier,
) -> None:
    initial_history_response = client_with_fake_classifier.get(
        "/api/v1/predictions",
        params={"limit": 20, "offset": 0},
    )
    initial_stats_response = client_with_fake_classifier.get("/api/v1/stats")

    initial_history = PredictionListResponse.model_validate(
        initial_history_response.json(),
    )
    initial_stats = StatsResponse.model_validate(initial_stats_response.json())

    assert initial_history_response.status_code == 200
    assert initial_stats_response.status_code == 200
    assert initial_history.total == 0
    assert initial_stats.total_predictions == 0
    assert initial_stats.count_by_label.praise == 0

    analyze_response = client_with_fake_classifier.post(
        "/api/v1/analyze",
        json={"text": "Спасибо за подробное объяснение и помощь"},
    )

    assert analyze_response.status_code == 200
    prediction_id = analyze_response.json()["id"]

    history_response = client_with_fake_classifier.get(
        "/api/v1/predictions",
        params={"limit": 20, "offset": 0},
    )
    stats_response = client_with_fake_classifier.get("/api/v1/stats")

    history = PredictionListResponse.model_validate(history_response.json())
    stats = StatsResponse.model_validate(stats_response.json())

    assert history_response.status_code == 200
    assert stats_response.status_code == 200
    assert history.total == 1
    assert len(history.items) == 1
    assert str(history.items[0].id) == prediction_id
    assert history.items[0].label == "praise"
    assert stats.total_predictions == 1
    assert stats.count_by_label.praise == 1
    assert stats.count_by_label.complaint == 0
    assert stats.average_confidence == 0.91
    assert stats.average_processing_time_ms is not None
    assert stats.last_prediction_at is not None
