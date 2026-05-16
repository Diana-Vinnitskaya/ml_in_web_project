from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.errors import (
    AppError,
    DependencyUnavailableError,
    InferenceError,
    ModelUnavailableError,
    PredictionNotFoundError,
    PersistenceError,
)
from app.core.logging import log_event, log_timing
from app.core.metrics import record_batch_prediction, record_prediction
from app.db.models import PredictionRecord
from app.db.repositories import (
    PredictionCreatePayload,
    PredictionPage,
    PredictionStats,
    create_prediction,
    create_predictions_batch,
    get_prediction_by_id as repository_get_prediction_by_id,
    get_prediction_stats as repository_get_prediction_stats,
    list_predictions as repository_list_predictions,
    update_predictions_processing_time,
    update_prediction_processing_time,
)
from app.db.session import check_database_availability, check_migration_state
from app.ml.model import FeedbackClassifier
from app.schemas.feedback import (
    AnalyzeRequest,
    BatchAnalyzeRequest,
    BatchPredictionResponse,
    CountByLabel,
    ErrorResponse,
    FeedbackLabel,
    HealthResponse,
    LivenessResponse,
    ModelInfoResponse,
    PredictionDetailResponse,
    PredictionListResponse,
    PredictionResponse,
    ProbabilityDistribution,
    ReadinessResponse,
    StatsResponse,
)


router = APIRouter()
logger = logging.getLogger(__name__)


VALIDATION_ERROR_RESPONSE = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "Request validation failed.",
    },
}
RATE_LIMITED_RESPONSE = {
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": ErrorResponse,
        "description": "Public API request rate exceeded at Nginx.",
    },
}
INTERNAL_ERROR_RESPONSE = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Unexpected inference, persistence, or service error.",
    },
}
SERVICE_UNAVAILABLE_RESPONSE = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Model, database, or migrations are not ready.",
    },
}
NOT_FOUND_RESPONSE = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "Resource not found.",
    },
}


@dataclass(slots=True)
class ReadinessState:
    model_loaded: bool
    database_available: bool
    migrations_applied: bool
    detail: str | None

    @property
    def ready(self) -> bool:
        return (
            self.model_loaded
            and self.database_available
            and self.migrations_applied
        )


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def get_classifier(request: Request) -> FeedbackClassifier:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None or not getattr(classifier, "loaded", False):
        raise ModelUnavailableError()
    return classifier


def get_session_factory(request: Request) -> sessionmaker[Session]:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise DependencyUnavailableError("Database session factory is unavailable")
    return session_factory


def get_db_session_dependency(request: Request) -> Iterator[Session]:
    session = get_session_factory(request)()
    try:
        yield session
    finally:
        session.close()


SettingsDependency = Annotated[Settings, Depends(get_app_settings)]
ClassifierDependency = Annotated[FeedbackClassifier, Depends(get_classifier)]
SessionDependency = Annotated[Session, Depends(get_db_session_dependency)]


def map_probability_distribution(
    probabilities: Mapping[str, float],
) -> ProbabilityDistribution:
    return ProbabilityDistribution(
        complaint=float(probabilities.get("complaint", 0.0)),
        question=float(probabilities.get("question", 0.0)),
        praise=float(probabilities.get("praise", 0.0)),
        other=float(probabilities.get("other", 0.0)),
    )


def map_prediction_response(record: PredictionRecord) -> PredictionResponse:
    return PredictionResponse(
        id=record.id,
        text=record.text,
        label=record.label,
        confidence=float(record.confidence),
        probabilities=map_probability_distribution(record.probabilities),
        processing_time_ms=float(record.processing_time_ms),
        created_at=record.created_at,
    )


def map_prediction_detail_response(
    record: PredictionRecord,
) -> PredictionDetailResponse:
    return PredictionDetailResponse(
        **map_prediction_response(record).model_dump(),
        model_name=record.model_name,
        model_version=record.model_version,
    )


def map_batch_prediction_response(
    records: list[PredictionRecord],
    *,
    processing_time_ms: float,
) -> BatchPredictionResponse:
    return BatchPredictionResponse(
        items=[map_prediction_response(record) for record in records],
        processing_time_ms=float(processing_time_ms),
    )


def map_prediction_list_response(
    page: PredictionPage,
    *,
    limit: int,
    offset: int,
) -> PredictionListResponse:
    return PredictionListResponse(
        items=[
            {
                "id": item.id,
                "text": item.text,
                "label": item.label,
                "confidence": float(item.confidence),
                "created_at": item.created_at,
            }
            for item in page.items
        ],
        limit=limit,
        offset=offset,
        total=page.total,
    )


def map_stats_response(stats: PredictionStats) -> StatsResponse:
    return StatsResponse(
        total_predictions=stats.total_predictions,
        count_by_label=CountByLabel(
            complaint=int(stats.count_by_label.get("complaint", 0)),
            question=int(stats.count_by_label.get("question", 0)),
            praise=int(stats.count_by_label.get("praise", 0)),
            other=int(stats.count_by_label.get("other", 0)),
        ),
        average_confidence=stats.average_confidence,
        average_processing_time_ms=stats.average_processing_time_ms,
        last_prediction_at=stats.last_prediction_at,
    )


def map_model_info_response(payload: Mapping[str, Any]) -> ModelInfoResponse:
    return ModelInfoResponse(**payload)


def get_model_info_payload(request: Request) -> dict[str, Any] | None:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        return None
    return classifier.get_info()


def build_readiness_state(request: Request) -> ReadinessState:
    session_factory = getattr(request.app.state, "session_factory", None)
    database_available, database_detail = check_database_availability(session_factory)
    migrations_applied, migration_detail = check_migration_state(session_factory)
    classifier = getattr(request.app.state, "classifier", None)
    model_loaded = bool(getattr(classifier, "loaded", False))

    request.app.state.database_available = database_available
    request.app.state.migrations_applied = migrations_applied
    request.app.state.model_loaded = model_loaded

    detail_parts: list[str] = []
    if not model_loaded:
        detail_parts.append(ModelUnavailableError.default_detail)
    if not database_available:
        detail_parts.append(database_detail or "Database is unavailable")
    elif not migrations_applied:
        detail_parts.append(migration_detail or "Migrations are not applied")

    detail = "; ".join(dict.fromkeys(detail_parts)) or None
    return ReadinessState(
        model_loaded=model_loaded,
        database_available=database_available,
        migrations_applied=migrations_applied,
        detail=detail,
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    tags=["Health"],
    summary="Liveness check",
    operation_id="getLiveness",
    responses={
        status.HTTP_200_OK: {
            "description": "Application process is alive.",
        },
    },
)
def get_liveness() -> LivenessResponse:
    return LivenessResponse(status="alive", service="backend")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness check",
    operation_id="getReadiness",
    responses={
        status.HTTP_200_OK: {
            "description": "Model, database, and migrations are ready.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "At least one dependency is unavailable.",
        },
    },
)
def get_readiness(
    request: Request,
    response: Response,
) -> ReadinessResponse:
    # API Health Check
    readiness = build_readiness_state(request)
    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="unavailable",
            model_loaded=readiness.model_loaded,
            database_available=readiness.database_available,
            migrations_applied=readiness.migrations_applied,
            detail=readiness.detail,
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=readiness.model_loaded,
        database_available=readiness.database_available,
        migrations_applied=readiness.migrations_applied,
        detail=None,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Aggregate UI-friendly health status",
    operation_id="getHealth",
    responses={
        status.HTTP_200_OK: {
            "description": "Aggregate health state.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "Service is alive but not ready.",
        },
    },
)
def get_health(
    request: Request,
    response: Response,
    settings: SettingsDependency,
) -> HealthResponse:
    readiness = build_readiness_state(request)
    model_info = get_model_info_payload(request)
    model_name = settings.model_name
    if model_info is not None:
        model_name = str(model_info.get("model_name") or model_name)

    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unavailable",
            model_loaded=readiness.model_loaded,
            database_available=readiness.database_available,
            model_name=model_name,
            detail=readiness.detail,
        )

    return HealthResponse(
        status="ok",
        model_loaded=readiness.model_loaded,
        database_available=readiness.database_available,
        model_name=model_name,
        detail=None,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model metadata, limits, labels, and metrics",
    operation_id="getModelInfo",
    responses={
        status.HTTP_200_OK: {
            "description": "Model profile.",
        },
        **SERVICE_UNAVAILABLE_RESPONSE,
    },
)
def get_model_info(
    classifier: ClassifierDependency,
) -> ModelInfoResponse:
    return map_model_info_response(classifier.get_info())


@router.post(
    "/analyze",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Classify one Russian feedback text",
    operation_id="analyzeFeedback",
    responses={
        status.HTTP_200_OK: {
            "description": "Saved prediction result.",
        },
        **VALIDATION_ERROR_RESPONSE,
        **RATE_LIMITED_RESPONSE,
        **INTERNAL_ERROR_RESPONSE,
        **SERVICE_UNAVAILABLE_RESPONSE,
    },
)
def analyze_feedback(
    payload: AnalyzeRequest,
    request: Request,
) -> PredictionResponse:
    total_started_at = perf_counter()
    predicted_label: str | None = None

    # Логирование
    log_event(
        logger,
        logging.INFO,
        "Single analysis request received",
        mode="single",
        text_length=len(payload.text),
    )

    try:
        settings = get_app_settings(request)
        classifier = get_classifier(request)
        session_factory = get_session_factory(request)
        model_info = classifier.get_info()
        model_name = str(model_info.get("model_name") or settings.model_name)
        model_version = str(model_info.get("version") or settings.model_version)

        preprocessing_started_at = perf_counter()
        text = payload.text.strip()
        log_timing(
            logger,
            "single_preprocessing",
            (perf_counter() - preprocessing_started_at) * 1000,
            mode="single",
            text_length=len(text),
        )

        inference_started_at = perf_counter()
        try:
            prediction = classifier.predict_one(text)
        except Exception as exc:  # pragma: no cover - defensive integration guard
            log_event(
                logger,
                logging.ERROR,
                "Single prediction inference failed",
                mode="single",
                error_type=type(exc).__name__,
            )
            raise InferenceError() from exc
        log_timing(
            logger,
            "single_inference",
            (perf_counter() - inference_started_at) * 1000,
            mode="single",
            label=prediction.label,
        )
        predicted_label = prediction.label

        with session_factory() as session:
            persistence_started_at = perf_counter()
            try:
                record = create_prediction(
                    session,
                    PredictionCreatePayload(
                        text=prediction.text,
                        label=prediction.label,
                        confidence=prediction.confidence,
                        probabilities=prediction.probabilities,
                        processing_time_ms=(
                            perf_counter() - total_started_at
                        )
                        * 1000,
                        model_name=model_name,
                        model_version=model_version,
                    ),
                )
                record = update_prediction_processing_time(
                    session,
                    record,
                    (perf_counter() - total_started_at) * 1000,
                )
            except SQLAlchemyError as exc:
                log_event(
                    logger,
                    logging.ERROR,
                    "Single prediction persistence failed",
                    mode="single",
                    error_type=type(exc).__name__,
                )
                raise PersistenceError() from exc

        log_timing(
            logger,
            "single_persistence",
            (perf_counter() - persistence_started_at) * 1000,
            mode="single",
            prediction_id=str(record.id),
        )
        record_prediction(
            mode="single",
            label=record.label,
            duration_ms=record.processing_time_ms,
            status="success",
        )
        log_timing(
            logger,
            "single_request_total",
            record.processing_time_ms,
            mode="single",
            status="success",
            prediction_id=str(record.id),
            label=record.label,
        )
        return map_prediction_response(record)
    except (DependencyUnavailableError, ModelUnavailableError) as exc:
        duration_ms = (perf_counter() - total_started_at) * 1000
        record_prediction(
            mode="single",
            label=predicted_label,
            duration_ms=duration_ms,
            status="unavailable",
        )
        log_timing(
            logger,
            "single_request_total",
            duration_ms,
            mode="single",
            status="unavailable",
            detail=str(exc.detail),
        )
        raise
    except AppError as exc:
        duration_ms = (perf_counter() - total_started_at) * 1000
        record_prediction(
            mode="single",
            label=predicted_label,
            duration_ms=duration_ms,
            status="error",
        )
        log_timing(
            logger,
            "single_request_total",
            duration_ms,
            mode="single",
            status="error",
            detail=str(exc.detail),
        )
        raise


@router.post(
    "/batch-analyze",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Classify a bounded batch of Russian feedback texts",
    operation_id="analyzeFeedbackBatch",
    responses={
        status.HTTP_200_OK: {
            "description": "Saved prediction results for every input item.",
        },
        **VALIDATION_ERROR_RESPONSE,
        **RATE_LIMITED_RESPONSE,
        **INTERNAL_ERROR_RESPONSE,
        **SERVICE_UNAVAILABLE_RESPONSE,
    },
)
def analyze_feedback_batch(
    payload: BatchAnalyzeRequest,
    request: Request,
) -> BatchPredictionResponse:
    total_started_at = perf_counter()
    predicted_labels: list[str] = []

    log_event(
        logger,
        logging.INFO,
        "Batch analysis request received",
        mode="batch",
        batch_size=len(payload.texts),
    )

    try:
        settings = get_app_settings(request)
        classifier = get_classifier(request)
        session_factory = get_session_factory(request)
        model_info = classifier.get_info()
        model_name = str(model_info.get("model_name") or settings.model_name)
        model_version = str(model_info.get("version") or settings.model_version)

        preprocessing_started_at = perf_counter()
        texts = [text.strip() for text in payload.texts]
        log_timing(
            logger,
            "batch_preprocessing",
            (perf_counter() - preprocessing_started_at) * 1000,
            mode="batch",
            batch_size=len(texts),
        )

        inference_started_at = perf_counter()
        try:
            predictions = classifier.predict_batch(texts)
        except Exception as exc:  # pragma: no cover - defensive integration guard
            log_event(
                logger,
                logging.ERROR,
                "Batch prediction inference failed",
                mode="batch",
                batch_size=len(texts),
                error_type=type(exc).__name__,
            )
            raise InferenceError() from exc
        if len(predictions) != len(texts):
            raise InferenceError("Batch inference returned an unexpected number of results")
        log_timing(
            logger,
            "batch_inference",
            (perf_counter() - inference_started_at) * 1000,
            mode="batch",
            batch_size=len(predictions),
        )
        predicted_labels = [prediction.label for prediction in predictions]

        with session_factory() as session:
            persistence_started_at = perf_counter()
            try:
                records = create_predictions_batch(
                    session,
                    [
                        PredictionCreatePayload(
                            text=prediction.text,
                            label=prediction.label,
                            confidence=prediction.confidence,
                            probabilities=prediction.probabilities,
                            processing_time_ms=0.0,
                            model_name=model_name,
                            model_version=model_version,
                        )
                        for prediction in predictions
                    ],
                )
                total_processing_time_ms = (perf_counter() - total_started_at) * 1000
                records = update_predictions_processing_time(
                    session,
                    records,
                    total_processing_time_ms,
                )
            except SQLAlchemyError as exc:
                log_event(
                    logger,
                    logging.ERROR,
                    "Batch prediction persistence failed",
                    mode="batch",
                    batch_size=len(texts),
                    error_type=type(exc).__name__,
                )
                raise PersistenceError() from exc

        log_timing(
            logger,
            "batch_persistence",
            (perf_counter() - persistence_started_at) * 1000,
            mode="batch",
            batch_size=len(records),
        )
        record_batch_prediction(
            labels=[record.label for record in records],
            duration_ms=total_processing_time_ms,
            status="success",
        )
        log_timing(
            logger,
            "batch_request_total",
            total_processing_time_ms,
            mode="batch",
            status="success",
            batch_size=len(records),
        )
        return map_batch_prediction_response(
            records,
            processing_time_ms=total_processing_time_ms,
        )
    except (DependencyUnavailableError, ModelUnavailableError) as exc:
        duration_ms = (perf_counter() - total_started_at) * 1000
        record_batch_prediction(
            labels=predicted_labels,
            duration_ms=duration_ms,
            status="unavailable",
        )
        log_timing(
            logger,
            "batch_request_total",
            duration_ms,
            mode="batch",
            status="unavailable",
            batch_size=len(payload.texts),
            detail=str(exc.detail),
        )
        raise
    except AppError as exc:
        duration_ms = (perf_counter() - total_started_at) * 1000
        record_batch_prediction(
            labels=predicted_labels,
            duration_ms=duration_ms,
            status="error",
        )
        log_timing(
            logger,
            "batch_request_total",
            duration_ms,
            mode="batch",
            status="error",
            batch_size=len(payload.texts),
            detail=str(exc.detail),
        )
        raise


@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    tags=["Predictions"],
    summary="List recent prediction history",
    operation_id="listPredictions",
    responses={
        status.HTTP_200_OK: {
            "description": "Bounded prediction history page.",
        },
        **VALIDATION_ERROR_RESPONSE,
        **SERVICE_UNAVAILABLE_RESPONSE,
    },
)
def get_predictions(
    session: SessionDependency,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    label: FeedbackLabel | None = Query(default=None),
) -> PredictionListResponse:
    try:
        page = repository_list_predictions(
            session,
            limit=limit,
            offset=offset,
            label=label.value if label is not None else None,
        )
    except SQLAlchemyError as exc:
        log_event(
            logger,
            logging.ERROR,
            "Prediction history query failed",
            limit=limit,
            offset=offset,
            label=label.value if label is not None else None,
            error_type=type(exc).__name__,
        )
        raise DependencyUnavailableError("Prediction history is unavailable") from exc

    return map_prediction_list_response(page, limit=limit, offset=offset)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailResponse,
    tags=["Predictions"],
    summary="Retrieve one prediction by id",
    operation_id="getPrediction",
    responses={
        status.HTTP_200_OK: {
            "description": "Full prediction detail.",
        },
        **NOT_FOUND_RESPONSE,
        **VALIDATION_ERROR_RESPONSE,
        **SERVICE_UNAVAILABLE_RESPONSE,
    },
)
def get_prediction(
    prediction_id: UUID,
    session: SessionDependency,
) -> PredictionDetailResponse:
    try:
        record = repository_get_prediction_by_id(session, prediction_id)
    except SQLAlchemyError as exc:
        log_event(
            logger,
            logging.ERROR,
            "Prediction detail query failed",
            prediction_id=str(prediction_id),
            error_type=type(exc).__name__,
        )
        raise DependencyUnavailableError("Prediction detail is unavailable") from exc

    if record is None:
        raise PredictionNotFoundError()

    return map_prediction_detail_response(record)


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Stats"],
    summary="Aggregate prediction statistics",
    operation_id="getPredictionStats",
    responses={
        status.HTTP_200_OK: {
            "description": "Aggregate statistics derived from prediction history.",
        },
        **SERVICE_UNAVAILABLE_RESPONSE,
    },
)
def get_stats(
    session: SessionDependency,
) -> StatsResponse:
    try:
        stats = repository_get_prediction_stats(session)
    except SQLAlchemyError as exc:
        log_event(
            logger,
            logging.ERROR,
            "Prediction statistics query failed",
            error_type=type(exc).__name__,
        )
        raise DependencyUnavailableError("Prediction statistics are unavailable") from exc

    return map_stats_response(stats)
