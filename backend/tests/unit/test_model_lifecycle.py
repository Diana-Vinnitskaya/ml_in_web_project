from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.ml.model import FeedbackClassifier
from app.ml.train import train_and_persist_model


def test_train_and_persist_model_writes_artifacts(
    app_settings: Settings,
) -> None:
    artifacts = train_and_persist_model(
        train_data_path=app_settings.train_data_path,
        model_path=app_settings.model_path,
        metrics_path=app_settings.model_metrics_path,
        model_name=app_settings.model_name,
        model_version=app_settings.model_version,
        labels=app_settings.labels,
    )

    metrics = json.loads(app_settings.model_metrics_path.read_text(encoding="utf-8"))

    assert app_settings.model_path.exists()
    assert app_settings.model_metrics_path.exists()
    assert tuple(artifacts.labels) == app_settings.labels
    assert metrics["accuracy"] >= 0.7
    assert metrics["macro_f1"] >= 0.7
    assert metrics["cross_validation"]["n_splits"] == 5
    assert metrics["cross_validation"]["n_repeats"] == 5
    assert metrics["cross_validation"]["accuracy_mean"] >= 0.7
    assert metrics["cross_validation"]["macro_f1_mean"] >= 0.7
    assert metrics["cross_validation"]["macro_f1_std"] >= 0.0
    assert metrics["train_size"] > 0
    assert metrics["test_size"] > 0
    assert "complaint" in metrics["classification_report"]
    assert "question" in metrics["classification_report"]
    assert "praise" in metrics["classification_report"]
    assert "other" in metrics["classification_report"]


def test_classifier_load_exposes_model_info(
    trained_classifier: FeedbackClassifier,
    app_settings: Settings,
) -> None:
    info = trained_classifier.get_info()

    assert trained_classifier.loaded is True
    assert info["model_name"] == app_settings.model_name
    assert info["version"] == app_settings.model_version
    assert tuple(info["labels"]) == app_settings.labels
    assert info["loaded"] is True
    assert info["metrics"]["accuracy"] >= 0.7
    assert info["metrics"]["macro_f1"] >= 0.7
    assert info["metrics"]["cross_validation"]["macro_f1_mean"] >= 0.7


def test_predict_one_returns_bounded_probabilities(
    trained_classifier: FeedbackClassifier,
    app_settings: Settings,
) -> None:
    result = trained_classifier.predict_one(
        "Поддержка не отвечает уже третий день и заказ не привезли",
    )

    assert result.label in app_settings.labels
    assert set(result.probabilities) == set(app_settings.labels)
    assert 0.0 <= result.confidence <= 1.0
    assert pytest.approx(sum(result.probabilities.values()), rel=1e-3, abs=1e-3) == 1.0
    for value in result.probabilities.values():
        assert 0.0 <= value <= 1.0


def test_predict_one_classifies_positive_app_feedback_as_praise(
    trained_classifier: FeedbackClassifier,
) -> None:
    result = trained_classifier.predict_one("Мне нравится приложение!")

    assert result.label == "praise"
    assert result.probabilities["praise"] > result.probabilities["complaint"]
    assert result.probabilities["praise"] > result.probabilities["question"]


def test_predict_batch_returns_one_result_per_input(
    trained_classifier: FeedbackClassifier,
    app_settings: Settings,
) -> None:
    texts = [
        "Спасибо за быструю доставку и аккуратную упаковку",
        "Когда откроется новый пункт выдачи рядом с домом",
        "Это просто информация о новом получателе заказа",
    ]

    results = trained_classifier.predict_batch(texts)

    assert len(results) == len(texts)
    assert [result.text for result in results] == texts
    for result in results:
        assert result.label in app_settings.labels
        assert set(result.probabilities) == set(app_settings.labels)
