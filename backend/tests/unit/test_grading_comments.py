from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("relative_path", "markers"),
    [
        (
            "backend/app/main.py",
            [
                "# Graceful Shutdown",
                "# Управление жизненным циклом контекстных переменных",
                "# Stateless архитектура",
                "# Метрики",
            ],
        ),
        (
            "backend/app/api/routes.py",
            [
                "# API Health Check",
                "# Логирование",
            ],
        ),
        (
            "backend/app/schemas/feedback.py",
            ["# Валидация данных"],
        ),
        (
            "backend/app/ml/model.py",
            [
                "# Изоляция ML-логики",
                "# Управление ресурсами",
            ],
        ),
        (
            "backend/app/db/models.py",
            ["# ORM"],
        ),
        (
            "docker-compose.yml",
            [
                "# Изоляция в сети",
                "# Разделение сетей",
                "# Управление постоянством данных",
                "# Compose Healthcheck",
                "# Умный depends_on",
                "# Порядок запуска",
            ],
        ),
        (
            "nginx/nginx.conf",
            [
                "# Единая точка входа",
                "# Маршрутизация",
                "# Rate Limiting",
            ],
        ),
        (
            "backend/Dockerfile",
            ["# Оптимизация сборки"],
        ),
        (
            "ui/Dockerfile",
            ["# Оптимизация сборки"],
        ),
        (
            "ui/app.py",
            [
                "# Слабая связность",
                "# Визуальная репрезентация",
            ],
        ),
    ],
)
def test_required_grading_comment_markers_are_present(
    relative_path: str,
    markers: list[str],
) -> None:
    target_path = REPO_ROOT / relative_path

    assert target_path.exists(), f"{relative_path} must exist."

    text = target_path.read_text(encoding="utf-8")

    for marker in markers:
        assert marker in text, f"{relative_path} is missing required marker: {marker}"
