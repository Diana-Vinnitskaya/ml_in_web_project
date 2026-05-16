from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
REQUIRED_TABLES = tuple(Base.metadata.tables.keys())


def _engine_options_for_url(database_url: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "future": True,
        "pool_pre_ping": not database_url.startswith("sqlite"),
    }
    if database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            options["poolclass"] = StaticPool
    return options


def create_engine_and_session_factory(
    *,
    settings: Settings | None = None,
    database_url: str | None = None,
) -> tuple[Engine, sessionmaker[Session]]:
    resolved_settings = settings or get_settings()
    resolved_database_url = database_url or resolved_settings.database_url
    if not resolved_database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    engine = create_engine(
        resolved_database_url,
        **_engine_options_for_url(resolved_database_url),
    )
    session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )
    return engine, session_factory


def init_session_factory(
    *,
    settings: Settings | None = None,
    database_url: str | None = None,
    force: bool = False,
) -> sessionmaker[Session]:
    global _engine, _session_factory

    if force or _engine is None or _session_factory is None:
        _engine, _session_factory = create_engine_and_session_factory(
            settings=settings,
            database_url=database_url,
        )

    return _session_factory


def configure_session_factory(
    session_factory: sessionmaker[Session],
    *,
    engine: Engine | None = None,
) -> None:
    global _engine, _session_factory

    _session_factory = session_factory
    _engine = engine or session_factory.kw.get("bind")


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        init_session_factory()
    if _engine is None:
        raise RuntimeError("Database engine is not initialized")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        init_session_factory()
    if _session_factory is None:
        raise RuntimeError("Session factory is not initialized")
    return _session_factory


def create_session() -> Session:
    return get_session_factory()()


def get_db_session() -> Iterator[Session]:
    session = create_session()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = create_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def is_database_available(session: Session | None = None) -> bool:
    if session is not None:
        session.execute(text("SELECT 1"))
        return True

    with session_scope() as active_session:
        active_session.execute(text("SELECT 1"))
    return True


def check_database_availability(
    session_factory: sessionmaker[Session] | None = None,
) -> tuple[bool, str | None]:
    resolved_session_factory = session_factory
    if resolved_session_factory is None:
        try:
            resolved_session_factory = get_session_factory()
        except Exception as exc:
            return False, str(exc)

    try:
        with resolved_session_factory() as session:
            is_database_available(session)
    except Exception as exc:
        return False, str(exc)

    return True, None


def check_migration_state(
    session_factory: sessionmaker[Session] | None = None,
    *,
    required_tables: Iterable[str] = REQUIRED_TABLES,
) -> tuple[bool, str | None]:
    resolved_session_factory = session_factory
    if resolved_session_factory is None:
        try:
            resolved_session_factory = get_session_factory()
        except Exception as exc:
            return False, str(exc)

    engine = resolved_session_factory.kw.get("bind")
    if not isinstance(engine, Engine):
        return False, "Session factory is not bound to an engine"

    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        missing_tables = sorted(set(required_tables) - existing_tables)
        if missing_tables:
            joined_tables = ", ".join(missing_tables)
            return False, f"Missing database tables: {joined_tables}"

        if str(engine.url).startswith("sqlite"):
            return True, None

        if "alembic_version" not in existing_tables:
            return False, "Alembic version table is missing"

        with engine.connect() as connection:
            version = connection.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1"),
            ).scalar_one_or_none()
    except Exception as exc:
        return False, str(exc)

    if version is None:
        return False, "Alembic revision is not recorded"

    return True, None


def dispose_engine() -> None:
    global _engine, _session_factory

    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
