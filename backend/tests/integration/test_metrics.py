from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

from app.core.metrics import metrics_content_type


def _read_sample_value(
    metrics_text: str,
    metric_name: str,
    labels: dict[str, str],
) -> float:
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name == metric_name and dict(sample.labels) == labels:
                return float(sample.value)
    return 0.0


def test_metrics_endpoint_exposes_prometheus_payload_and_prediction_increments(
    client_with_fake_classifier,
) -> None:
    before_metrics = client_with_fake_classifier.get("/metrics")

    assert before_metrics.status_code == 200
    assert before_metrics.headers["content-type"] == metrics_content_type()
    assert "rufeedback_predictions_total" in before_metrics.text
    assert "rufeedback_http_requests_total" in before_metrics.text

    before_http_total = _read_sample_value(
        before_metrics.text,
        "rufeedback_http_requests_total",
        {
            "method": "POST",
            "path": "/api/v1/analyze",
            "status": "200",
        },
    )
    before_prediction_total = _read_sample_value(
        before_metrics.text,
        "rufeedback_predictions_total",
        {
            "mode": "single",
            "status": "success",
            "label": "praise",
        },
    )
    before_prediction_duration_count = _read_sample_value(
        before_metrics.text,
        "rufeedback_prediction_duration_seconds_count",
        {
            "mode": "single",
            "status": "success",
        },
    )

    analyze_response = client_with_fake_classifier.post(
        "/api/v1/analyze",
        json={"text": "Спасибо за понятную консультацию и помощь"},
    )

    assert analyze_response.status_code == 200

    after_metrics = client_with_fake_classifier.get("/metrics")

    assert after_metrics.status_code == 200

    after_http_total = _read_sample_value(
        after_metrics.text,
        "rufeedback_http_requests_total",
        {
            "method": "POST",
            "path": "/api/v1/analyze",
            "status": "200",
        },
    )
    after_prediction_total = _read_sample_value(
        after_metrics.text,
        "rufeedback_predictions_total",
        {
            "mode": "single",
            "status": "success",
            "label": "praise",
        },
    )
    after_prediction_duration_count = _read_sample_value(
        after_metrics.text,
        "rufeedback_prediction_duration_seconds_count",
        {
            "mode": "single",
            "status": "success",
        },
    )

    assert after_http_total == before_http_total + 1.0
    assert after_prediction_total == before_prediction_total + 1.0
    assert after_prediction_duration_count == before_prediction_duration_count + 1.0
