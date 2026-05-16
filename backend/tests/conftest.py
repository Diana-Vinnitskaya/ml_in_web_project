from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.errors import ModelUnavailableError
from app.db.base import Base
from app.db.session import create_engine_and_session_factory
from app.main import create_app
from app.ml.model import FeedbackClassifier, PredictionResult


class FakeClassifier:
    def __init__(self, settings: Settings, *, loaded: bool = True) -> None:
        self.settings = settings
        self.loaded = loaded
        self._load_succeeds = loaded

    def train_if_missing(self) -> None:
        return None

    def load(self) -> None:
        if not self._load_succeeds:
            raise ModelUnavailableError()
        self.loaded = True

    def predict_one(self, text: str) -> PredictionResult:
        return PredictionResult(
            text=text,
            label="praise",
            confidence=0.91,
            probabilities={
                "complaint": 0.02,
                "question": 0.02,
                "praise": 0.91,
                "other": 0.05,
            },
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict_one(text) for text in texts]

    def get_info(self) -> dict[str, object]:
        return {
            "model_name": self.settings.model_name,
            "version": self.settings.model_version,
            "labels": list(self.settings.labels),
            "max_text_length": self.settings.max_text_length,
            "max_batch_size": self.settings.max_batch_size,
            "loaded": self.loaded,
            "metrics": {
                "accuracy": 0.95,
                "macro_f1": 0.95,
            },
        }


@pytest.fixture
def repo_train_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "train.csv"


@pytest.fixture
def app_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_train_data_path: Path,
) -> Iterator[Settings]:
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    database_path = tmp_path / "test.db"

    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "feedback_classifier.joblib"))
    monkeypatch.setenv("MODEL_METRICS_PATH", str(model_dir / "metrics.json"))
    monkeypatch.setenv("TRAIN_DATA_PATH", str(repo_train_data_path))
    monkeypatch.setenv("PROJECT_NAME", "RuFeedback Classifier Test")
    monkeypatch.setenv("API_PREFIX", "/api/v1")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    get_settings.cache_clear()

    try:
        yield Settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
def session_factory(app_settings: Settings) -> Iterator[sessionmaker[Session]]:
    engine, factory = create_engine_and_session_factory(settings=app_settings)
    Base.metadata.create_all(engine)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def classifier(app_settings: Settings) -> FeedbackClassifier:
    return FeedbackClassifier(settings=app_settings)


@pytest.fixture
def trained_classifier(classifier: FeedbackClassifier) -> FeedbackClassifier:
    classifier.train_if_missing()
    classifier.load()
    return classifier


@pytest.fixture
def fake_classifier(app_settings: Settings) -> FakeClassifier:
    return FakeClassifier(app_settings)


@pytest.fixture
def unavailable_classifier(app_settings: Settings) -> FakeClassifier:
    return FakeClassifier(app_settings, loaded=False)


@pytest.fixture
def client(
    app_settings: Settings,
    session_factory: sessionmaker[Session],
) -> Iterator[TestClient]:
    with TestClient(
        create_app(
            settings=app_settings,
            session_factory=session_factory,
        ),
    ) as test_client:
        yield test_client


@pytest.fixture
def client_with_fake_classifier(
    app_settings: Settings,
    session_factory: sessionmaker[Session],
    fake_classifier: FakeClassifier,
) -> Iterator[TestClient]:
    with TestClient(
        create_app(
            settings=app_settings,
            classifier=fake_classifier,
            session_factory=session_factory,
        ),
    ) as test_client:
        yield test_client


@pytest.fixture
def client_with_unavailable_classifier(
    app_settings: Settings,
    session_factory: sessionmaker[Session],
    unavailable_classifier: FakeClassifier,
) -> Iterator[TestClient]:
    with TestClient(
        create_app(
            settings=app_settings,
            classifier=unavailable_classifier,
            session_factory=session_factory,
        ),
    ) as test_client:
        yield test_client
