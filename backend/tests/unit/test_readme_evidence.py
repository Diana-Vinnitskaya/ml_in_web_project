from __future__ import annotations

from pathlib import Path

import pytest


README_PATH = Path(__file__).resolve().parents[3] / "README.md"


def read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "section",
    [
        "## Назначение сервиса",
        "## Архитектура",
        "## Технологический стек",
        "## Локальные ограничения",
        "## Быстрый старт",
        "## Примеры API",
        "## Обучение модели и артефакты",
        "## База данных и миграции",
        "## Проверки здоровья и мониторинг",
        "## Устранение неполадок",
        "## Покрытие критериев оценки",
        "## Выбранные дополнительные пункты",
    ],
)
def test_readme_contains_required_sections(section: str) -> None:
    text = read_readme()

    assert README_PATH.exists(), "README.md must exist at the repository root."
    assert section in text


@pytest.mark.parametrize(
    "theme",
    [
        "1. API-бэкенд",
        "2. ML-сервис",
        "3. Интерфейс",
        "4. Реверс-прокси",
        "5. Работа с данными и состоянием",
        "6. Оркестрация Docker Compose",
        "7. Отказоустойчивость",
        "8. Проверки здоровья и мониторинг",
    ],
)
def test_readme_lists_all_mandatory_themes(theme: str) -> None:
    assert theme in read_readme()


@pytest.mark.parametrize(
    "extra_item",
    [
        "EX-001",
        "Своя модель (+5)",
        "EX-002",
        "Визуальная репрезентация (+5)",
        "EX-003",
        "Метрики Prometheus + Grafana (+3)",
        "EX-004",
        "Кэширование слоев Dockerfile (+1 желательно)",
        "EX-005",
        "Менеджмент зависимостей приложения (+1 желательно)",
    ],
)
def test_readme_lists_selected_extra_items(extra_item: str) -> None:
    assert extra_item in read_readme()


def test_readme_mentions_required_operational_evidence() -> None:
    text = read_readme()

    required_fragments = [
        "cp .env.example .env",
        "docker compose up --build -d",
        "http://localhost/",
        "http://localhost/grafana/",
        "POST /api/v1/analyze",
        "POST /api/v1/batch-analyze",
        "GET /api/v1/health/live",
        "GET /api/v1/health/ready",
        "GET /api/v1/health",
        "GET /metrics",
        "/app/models/feedback_classifier.joblib",
        "/app/models/metrics.json",
        "docker compose exec backend python -m app.ml.train",
        "docker compose exec backend alembic upgrade head",
        "scripts/smoke_compose.sh",
        "Браузер пользователя",
        "Nginx :80",
        "Бэкенд FastAPI :8000",
        "Интерфейс Streamlit :8501",
        "PostgreSQL :5432",
    ]

    for fragment in required_fragments:
        assert fragment in text


def test_readme_mentions_required_troubleshooting_steps() -> None:
    text = read_readme()

    required_fragments = [
        "порт 80 уже занят",
        "NGINX_PORT",
        "docker compose logs backend",
        "docker compose logs postgres",
        "docker compose down -v",
        "docker compose up --build -d",
    ]

    for fragment in required_fragments:
        assert fragment in text
