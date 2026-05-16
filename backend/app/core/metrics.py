from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


PREDICTION_LABELS: Final[tuple[str, str, str, str]] = (
    "complaint",
    "question",
    "praise",
    "other",
)
UNKNOWN_LABEL: Final[str] = "unknown"

# Метрики: общие HTTP- и prediction-метрики используются всеми story-слоями.
HTTP_REQUESTS_TOTAL = Counter(
    "rufeedback_http_requests_total",
    "Total HTTP requests served by the backend.",
    labelnames=("method", "path", "status"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "rufeedback_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
PREDICTIONS_TOTAL = Counter(
    "rufeedback_predictions_total",
    "Total prediction results grouped by request mode, status, and predicted label.",
    labelnames=("mode", "status", "label"),
)
PREDICTION_DURATION_SECONDS = Histogram(
    "rufeedback_prediction_duration_seconds",
    "Prediction request duration in seconds grouped by request mode and status.",
    labelnames=("mode", "status"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)


def normalize_prediction_label(label: str | None) -> str:
    if not label:
        return UNKNOWN_LABEL

    normalized = label.strip().lower()
    if normalized in PREDICTION_LABELS:
        return normalized
    return UNKNOWN_LABEL


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    return normalized or "unknown"


def _seconds_from_milliseconds(duration_ms: float) -> float:
    return max(float(duration_ms), 0.0) / 1000.0


def observe_http_request(
    *,
    method: str,
    path: str,
    status_code: int | str,
    duration_ms: float,
) -> None:
    normalized_method = method.strip().upper() or "UNKNOWN"
    normalized_path = path.strip() or "/"
    normalized_status = str(status_code)

    HTTP_REQUESTS_TOTAL.labels(
        method=normalized_method,
        path=normalized_path,
        status=normalized_status,
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=normalized_method,
        path=normalized_path,
    ).observe(_seconds_from_milliseconds(duration_ms))


def record_prediction(
    *,
    mode: str,
    label: str | None,
    duration_ms: float,
    status: str = "success",
) -> None:
    normalized_mode = normalize_mode(mode)
    normalized_status = status.strip().lower() or "unknown"
    normalized_label = normalize_prediction_label(label)

    PREDICTIONS_TOTAL.labels(
        mode=normalized_mode,
        status=normalized_status,
        label=normalized_label,
    ).inc()
    PREDICTION_DURATION_SECONDS.labels(
        mode=normalized_mode,
        status=normalized_status,
    ).observe(_seconds_from_milliseconds(duration_ms))


def record_batch_prediction(
    *,
    labels: Iterable[str],
    duration_ms: float,
    status: str = "success",
) -> None:
    normalized_labels = [normalize_prediction_label(label) for label in labels]
    if not normalized_labels:
        normalized_labels = [UNKNOWN_LABEL]

    for label in normalized_labels:
        PREDICTIONS_TOTAL.labels(
            mode="batch",
            status=status.strip().lower() or "unknown",
            label=label,
        ).inc()

    PREDICTION_DURATION_SECONDS.labels(
        mode="batch",
        status=status.strip().lower() or "unknown",
    ).observe(_seconds_from_milliseconds(duration_ms))


def get_metrics_registry() -> CollectorRegistry:
    return REGISTRY


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


def render_metrics(registry: CollectorRegistry = REGISTRY) -> bytes:
    return generate_latest(registry)
