from __future__ import annotations

import json
from dataclasses import dataclass
from threading import RLock
from typing import Any, Sequence

import joblib
from sklearn.pipeline import Pipeline

from app.core.config import Settings, get_settings
from app.core.errors import ModelUnavailableError
from app.ml.train import train_and_persist_model


@dataclass(slots=True)
class PredictionResult:
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class FeedbackClassifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_path = self.settings.model_path
        self.metrics_path = self.settings.model_metrics_path
        self.train_data_path = self.settings.train_data_path

        self._lock = RLock()
        self._pipeline: Pipeline | None = None
        self._metrics: dict[str, Any] | None = None
        self._loaded = False
        self._labels = tuple(self.settings.labels)
        self._model_name = self.settings.model_name
        self._model_version = self.settings.model_version

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def train_if_missing(self, *, force: bool = False) -> None:
        with self._lock:
            if (
                not force
                and self.model_path.exists()
                and self.metrics_path.exists()
            ):
                return

            train_and_persist_model(
                train_data_path=self.train_data_path,
                model_path=self.model_path,
                metrics_path=self.metrics_path,
                model_name=self._model_name,
                model_version=self._model_version,
                labels=self.settings.labels,
            )

    def load(self, *, force: bool = False) -> None:
        with self._lock:
            if self._loaded and self._pipeline is not None and not force:
                return

            if force or not self.model_path.exists() or not self.metrics_path.exists():
                self.train_if_missing(force=force)

            bundle = joblib.load(self.model_path)
            pipeline = bundle.get("pipeline")
            if pipeline is None:
                raise ValueError(f"Model artifact does not contain a pipeline: {self.model_path}")

            self._pipeline = pipeline
            self._labels = tuple(bundle.get("labels") or self.settings.labels)
            self._model_name = str(bundle.get("model_name") or self.settings.model_name)
            self._model_version = str(
                bundle.get("model_version") or self.settings.model_version,
            )
            self._metrics = self._load_metrics()
            self._loaded = True

    def _load_metrics(self) -> dict[str, Any] | None:
        if not self.metrics_path.exists():
            return None
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))

    def _require_pipeline(self) -> Pipeline:
        if not self._loaded or self._pipeline is None:
            raise ModelUnavailableError()
        return self._pipeline

    def _collect_probabilities(
        self,
        texts: Sequence[str],
    ) -> list[PredictionResult]:
        pipeline = self._require_pipeline()
        probability_rows = pipeline.predict_proba(list(texts))
        classes = [str(label) for label in getattr(pipeline, "classes_", [])]
        if not classes:
            classifier = pipeline.named_steps.get("classifier")
            classes = [str(label) for label in getattr(classifier, "classes_", [])]

        results: list[PredictionResult] = []
        for text, row in zip(texts, probability_rows):
            probabilities = {label: 0.0 for label in self.settings.labels}
            for label, value in zip(classes, row):
                probabilities[str(label)] = float(value)

            top_label = max(probabilities, key=probabilities.get)
            results.append(
                PredictionResult(
                    text=text,
                    label=top_label,
                    confidence=float(probabilities[top_label]),
                    probabilities=probabilities,
                ),
            )
        return results

    def predict_one(self, text: str) -> PredictionResult:
        # Изоляция ML-логики
        return self._collect_probabilities([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        if not texts:
            return []
        return self._collect_probabilities(texts)

    def get_info(self) -> dict[str, Any]:
        metrics = self._metrics
        if metrics is None and self.metrics_path.exists():
            metrics = self._load_metrics()

        return {
            "model_name": self._model_name,
            "version": self._model_version,
            "labels": list(self._labels),
            "max_text_length": self.settings.max_text_length,
            "max_batch_size": self.settings.max_batch_size,
            "loaded": self.loaded,
            "metrics": metrics,
        }

    # Управление ресурсами
    def reset(self) -> None:
        with self._lock:
            self._pipeline = None
            self._metrics = None
            self._loaded = False
