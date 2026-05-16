from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4


request_id_context: ContextVar[str | None] = ContextVar(
    "request_id_context",
    default=None,
)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        structured_data = getattr(record, "structured_data", None)
        if not structured_data:
            return rendered

        payload = json.dumps(
            structured_data,
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )
        return f"{rendered} | {payload}"


def configure_logging(log_level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    if getattr(root_logger, "_rufeedback_logging_configured", False):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        StructuredFormatter(
            "%(asctime)s %(levelname)s [%(name)s] request_id=%(request_id)s %(message)s",
        )
    )
    handler.addFilter(RequestIdFilter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger._rufeedback_logging_configured = True


def bind_request_id(request_id: str | None = None) -> Token[str | None]:
    return request_id_context.set(request_id or make_request_id())


def clear_request_id(token: Token[str | None] | None = None) -> None:
    if token is None:
        request_id_context.set(None)
        return
    request_id_context.reset(token)


def get_request_id() -> str | None:
    return request_id_context.get()


def make_request_id() -> str:
    return uuid4().hex


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **structured_data: Any,
) -> None:
    logger.log(level, message, extra={"structured_data": structured_data})


def log_timing(
    logger: logging.Logger,
    stage: str,
    duration_ms: float,
    **structured_data: Any,
) -> None:
    log_event(
        logger,
        logging.INFO,
        f"{stage} completed",
        stage=stage,
        duration_ms=round(duration_ms, 3),
        **structured_data,
    )


@contextmanager
def timed_stage(
    logger: logging.Logger,
    stage: str,
    **structured_data: Any,
) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        duration_ms = (perf_counter() - start) * 1000
        log_timing(logger, stage, duration_ms, **structured_data)
