from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_request_id, log_event


logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "Application error"

    def __init__(self, detail: str | list[dict[str, Any]] | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(str(self.detail))


class DependencyUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Dependency is unavailable"


class ModelUnavailableError(DependencyUnavailableError):
    default_detail = "Model is not loaded"


class PredictionNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = "Prediction not found"


class PersistenceError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "Failed to persist prediction"


class InferenceError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "Inference failed"


def error_payload(detail: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "detail": jsonable_encoder(
            detail,
            custom_encoder={BaseException: str},
        ),
    }
    request_id = get_request_id()
    if request_id:
        payload["request_id"] = request_id
    return payload


def error_response(status_code: int, detail: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(detail))


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    log_event(
        logger,
        logging.WARNING if exc.status_code < 500 else logging.ERROR,
        "Application error handled",
        status_code=exc.status_code,
        detail=exc.detail,
        error_type=type(exc).__name__,
    )
    return error_response(exc.status_code, exc.detail)


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if exc.detail is not None else "HTTP error"
    log_event(
        logger,
        logging.WARNING if exc.status_code < 500 else logging.ERROR,
        "HTTP exception handled",
        status_code=exc.status_code,
        detail=detail,
    )
    return error_response(exc.status_code, detail)


async def validation_exception_handler(
    _: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    detail = exc.errors()
    log_event(
        logger,
        logging.WARNING,
        "Validation error handled",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        errors=detail,
    )
    return error_response(status.HTTP_422_UNPROCESSABLE_CONTENT, detail)


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    log_event(
        logger,
        logging.ERROR,
        "Unhandled exception handled",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(exc),
        error_type=type(exc).__name__,
    )
    return error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error")


def install_exception_handlers(app: FastAPI) -> None:
    # Обработка ошибок
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
