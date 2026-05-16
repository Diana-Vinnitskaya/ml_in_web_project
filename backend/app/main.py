from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import router as api_router
from app.core.config import Settings, get_settings
from app.core.errors import install_exception_handlers
from app.core.logging import (
    bind_request_id,
    clear_request_id,
    configure_logging,
    get_request_id,
    log_event,
)
from app.core.metrics import (
    metrics_content_type,
    observe_http_request,
    render_metrics,
)
from app.db.base import Base
from app.db.session import (
    configure_session_factory,
    create_engine_and_session_factory,
    dispose_engine,
    is_database_available,
)
from app.ml.model import FeedbackClassifier


logger = logging.getLogger(__name__)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_alembic_config(settings: Settings) -> AlembicConfig:
    backend_root = _backend_root()
    config = AlembicConfig(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    return config


def _prepare_database(
    settings: Settings,
    *,
    provided_session_factory: sessionmaker[Session] | None = None,
) -> tuple[sessionmaker[Session] | None, bool, bool, str | None]:
    detail: str | None = None
    database_available = False
    migrations_applied = False
    session_factory = provided_session_factory

    try:
        if provided_session_factory is None:
            engine, session_factory = create_engine_and_session_factory(settings=settings)
        else:
            session_factory = provided_session_factory
            engine = session_factory.kw.get("bind")
            if not isinstance(engine, Engine):
                raise RuntimeError("Session factory is not bound to an engine.")

        configure_session_factory(session_factory, engine=engine)

        with session_factory() as session:
            is_database_available(session)
        database_available = True

        if str(engine.url).startswith("sqlite"):
            Base.metadata.create_all(engine)
        else:
            command.upgrade(_make_alembic_config(settings), "head")
        migrations_applied = True
        return session_factory, database_available, migrations_applied, None
    except Exception as exc:  # pragma: no cover - startup safety net
        detail = str(exc)
        log_event(
            logger,
            logging.ERROR,
            "Database startup failed",
            detail=detail,
        )
        return session_factory, database_available, migrations_applied, detail


def _prepare_classifier(
    settings: Settings,
    *,
    provided_classifier: FeedbackClassifier | None = None,
) -> tuple[FeedbackClassifier, bool, str | None]:
    classifier = provided_classifier or FeedbackClassifier(settings=settings)
    try:
        classifier.train_if_missing()
        classifier.load()
        return classifier, classifier.loaded, None
    except Exception as exc:  # pragma: no cover - startup safety net
        detail = str(exc)
        log_event(
            logger,
            logging.ERROR,
            "Classifier startup failed",
            detail=detail,
        )
        return classifier, False, detail


def create_app(
    *,
    settings: Settings | None = None,
    classifier: FeedbackClassifier | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    app_settings = settings
    api_prefix = app_settings.api_prefix if app_settings is not None else "/api/v1"
    openapi_title = "RuFeedback Classifier API"
    openapi_servers = [
        {
            "url": api_prefix,
            "description": "Public API path behind Nginx",
        },
    ]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_settings = app_settings or get_settings()
        configure_logging(resolved_settings.log_level)

        app.state.settings = resolved_settings
        app.state.session_factory = None
        app.state.classifier = classifier or FeedbackClassifier(settings=resolved_settings)
        app.state.database_available = False
        app.state.migrations_applied = False
        app.state.model_loaded = False
        app.state.startup_errors = []

        resolved_session_factory, database_available, migrations_applied, db_detail = (
            _prepare_database(
                resolved_settings,
                provided_session_factory=session_factory,
            )
        )
        app.state.session_factory = resolved_session_factory
        app.state.database_available = database_available
        app.state.migrations_applied = migrations_applied
        if db_detail:
            app.state.startup_errors.append(db_detail)

        resolved_classifier, model_loaded, classifier_detail = _prepare_classifier(
            resolved_settings,
            provided_classifier=classifier,
        )
        app.state.classifier = resolved_classifier
        app.state.model_loaded = model_loaded
        if classifier_detail:
            app.state.startup_errors.append(classifier_detail)

        log_event(
            logger,
            logging.INFO,
            "Application startup completed",
            database_available=database_available,
            migrations_applied=migrations_applied,
            model_loaded=model_loaded,
        )

        try:
            yield
        finally:
            # Graceful Shutdown
            dispose_engine()
            app.state.database_available = False
            app.state.migrations_applied = False
            app.state.model_loaded = False

    app = FastAPI(
        title=openapi_title,
        version="0.1.0",
        description="REST contract for the local Russian feedback classification service.",
        lifespan=lifespan,
        servers=openapi_servers,
    )
    install_exception_handlers(app)
    app.include_router(api_router, prefix=api_prefix)

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            servers=openapi_servers,
        )

        public_paths: dict[str, Any] = {}
        for path, path_item in schema["paths"].items():
            if path.startswith(api_prefix) and path != api_prefix:
                public_paths[path.removeprefix(api_prefix)] = path_item
            else:
                public_paths[path] = path_item

        schema["paths"] = public_paths
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next,
    ) -> Response:
        # Управление жизненным циклом контекстных переменных
        token = bind_request_id(request.headers.get("X-Request-ID"))
        started_at = perf_counter()
        response: Response | None = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            observe_http_request(
                method=request.method,
                path=path,
                status_code=status_code,
                duration_ms=(perf_counter() - started_at) * 1000,
            )

            request_id = get_request_id()
            if response is not None and request_id:
                response.headers["X-Request-ID"] = request_id
            clear_request_id(token)

    @app.get("/__setup__", include_in_schema=False)
    def setup_status() -> dict[str, Any]:
        # Stateless архитектура
        return {
            "status": "setup",
            "api_prefix": api_prefix,
            "detail": "Foundation lifecycle and shared dependencies are configured.",
        }

    @app.get(
        "/metrics",
        tags=["Stats"],
        summary="Prometheus metrics endpoint",
        operation_id="getPrometheusMetrics",
        response_class=PlainTextResponse,
        responses={
            200: {
                "description": "Prometheus text exposition format.",
                "content": {
                    "text/plain": {
                        "schema": {
                            "type": "string",
                        },
                    },
                },
            },
        },
        openapi_extra={
            "servers": [
                {
                    "url": "/",
                },
            ],
        },
    )
    def metrics_endpoint() -> Response:
        # Метрики
        return PlainTextResponse(
            content=render_metrics(),
            media_type=metrics_content_type(),
        )

    return app


app = create_app()
