from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from app.core.config import Settings, get_settings


DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.25
DEFAULT_MAX_FEATURES = 5000
DEFAULT_CV_SPLITS = 5
DEFAULT_CV_REPEATS = 5
REQUIRED_COLUMNS = {"text", "label"}


@dataclass(slots=True)
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict[str, Any]
    labels: tuple[str, ...]


def _json_ready(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=float,
        ),
    )


def load_training_frame(
    train_data_path: Path,
    *,
    labels: tuple[str, ...],
) -> pd.DataFrame:
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data was not found: {train_data_path}")

    frame = pd.read_csv(train_data_path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Training data is missing required columns: "
            f"{', '.join(sorted(missing_columns))}",
        )

    prepared = frame.loc[:, ["text", "label"]].copy()
    prepared["text"] = prepared["text"].fillna("").astype(str).str.strip()
    prepared["label"] = prepared["label"].fillna("").astype(str).str.strip()
    prepared = prepared[(prepared["text"] != "") & (prepared["label"] != "")]

    unexpected_labels = sorted(set(prepared["label"]) - set(labels))
    if unexpected_labels:
        raise ValueError(
            "Training data contains unsupported labels: "
            f"{', '.join(unexpected_labels)}",
        )

    for label in labels:
        label_count = int((prepared["label"] == label).sum())
        if label_count < 2:
            raise ValueError(
                f"Training data must contain at least 2 rows for label '{label}'.",
            )

    return prepared.reset_index(drop=True)


def build_pipeline(*, max_features: int = DEFAULT_MAX_FEATURES) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=max_features,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=DEFAULT_RANDOM_STATE,
                ),
            ),
        ],
    )


def summarize_cross_validation(
    *,
    features: list[str],
    target: list[str],
    max_features: int,
    random_state: int,
    cv_splits: int = DEFAULT_CV_SPLITS,
    cv_repeats: int = DEFAULT_CV_REPEATS,
) -> dict[str, Any]:
    pipeline = build_pipeline(max_features=max_features)
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=None,
    )

    accuracy_scores = scores["test_accuracy"]
    macro_f1_scores = scores["test_macro_f1"]
    return {
        "n_splits": cv_splits,
        "n_repeats": cv_repeats,
        "accuracy_mean": float(accuracy_scores.mean()),
        "accuracy_std": float(accuracy_scores.std()),
        "accuracy_min": float(accuracy_scores.min()),
        "accuracy_max": float(accuracy_scores.max()),
        "macro_f1_mean": float(macro_f1_scores.mean()),
        "macro_f1_std": float(macro_f1_scores.std()),
        "macro_f1_min": float(macro_f1_scores.min()),
        "macro_f1_max": float(macro_f1_scores.max()),
    }


def train_and_persist_model(
    *,
    train_data_path: Path,
    model_path: Path,
    metrics_path: Path,
    model_name: str,
    model_version: str,
    labels: tuple[str, ...],
    max_features: int = DEFAULT_MAX_FEATURES,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainingArtifacts:
    frame = load_training_frame(train_data_path, labels=labels)
    features = frame["text"].tolist()
    target = frame["label"].tolist()
    cross_validation = summarize_cross_validation(
        features=features,
        target=target,
        max_features=max_features,
        random_state=random_state,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    pipeline = build_pipeline(max_features=max_features)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    report = _json_ready(
        classification_report(
            y_test,
            predictions,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
    )

    trained_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "cross_validation": cross_validation,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "trained_at": trained_at,
        "model_name": model_name,
        "version": model_version,
        "labels": list(labels),
        "vectorizer": {
            "max_features": max_features,
            "ngram_range": [1, 2],
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "labels": list(labels),
            "model_name": model_name,
            "model_version": model_version,
            "trained_at": trained_at,
        },
        model_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return TrainingArtifacts(pipeline=pipeline, metrics=metrics, labels=labels)


def train_from_settings(settings: Settings | None = None) -> TrainingArtifacts:
    resolved_settings = settings or get_settings()
    return train_and_persist_model(
        train_data_path=resolved_settings.train_data_path,
        model_path=resolved_settings.model_path,
        metrics_path=resolved_settings.model_metrics_path,
        model_name=resolved_settings.model_name,
        model_version=resolved_settings.model_version,
        labels=resolved_settings.labels,
    )


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the RuFeedback classifier.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=settings.train_data_path,
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.model_path,
        help="Path where the trained model artifact will be written.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=settings.model_metrics_path,
        help="Path where training metrics JSON will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    train_and_persist_model(
        train_data_path=args.train_data,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        model_name=settings.model_name,
        model_version=settings.model_version,
        labels=settings.labels,
    )


if __name__ == "__main__":
    main()
