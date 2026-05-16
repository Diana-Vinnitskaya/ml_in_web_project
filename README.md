# RuFeedback Classifier

RuFeedback Classifier — локальный ML-веб-сервис для классификации русскоязычной обратной связи по категориям `complaint`, `question`, `praise` и `other`. FastAPI отвечает за REST-бэкенд, Streamlit — за интерфейс, PostgreSQL — за хранение истории предсказаний, Nginx — за единую публичную точку входа, а Prometheus/Grafana — метрики.

Сервис возвращает предсказанную метку, уверенность модели, вероятности по всем классам, время обработки и идентификатор сохраненного предсказания. Модель — это собственный `scikit-learn`-пайплайн на базе `TfidfVectorizer + LogisticRegression`, обучался на `backend/data/train.csv` и сохраняется в `model_data`.

## Назначение сервиса

- Классифицировать один текст отзыва через `POST /api/v1/analyze`.
- Классифицировать до 32 отзывов в одном запросе через `POST /api/v1/batch-analyze`.
- Сохранять каждое предсказание в PostgreSQL.
- Показывать статусы здоровья, метаданные модели, историю и агрегированную статистику в интерфейсе Streamlit.
- Давать проверки состояния, логи, метрики Prometheus и дашборды Grafana.

## Архитектура

```mermaid
flowchart TB
    User["Браузер пользователя"] --> Nginx["Nginx :80"]
    Nginx --> UI["Интерфейс Streamlit :8501"]
    Nginx -->|"/api/v1/*"| Backend["Бэкенд FastAPI :8000"]
    Backend --> Classifier["FeedbackClassifier\nTF-IDF + LogisticRegression"]
    Backend --> Postgres["PostgreSQL :5432"]
    Prometheus -->|"/api/v1/metrics"| Backend
    Grafana["Grafana"] --> Prometheus
```

Внешний доступ ограничен адресом `http://localhost/`. Порты бэкенда `8000`, интерфейса `8501`, PostgreSQL `5432`, Prometheus `9090` и Grafana `3000` остаются внутренними для Docker Compose и доступны только через сервисные сети или реверс-прокси Nginx.

## Технологический стек

- Бэкенд: FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic, psycopg, Python 3.11.
- ML-часть: `scikit-learn`, `TfidfVectorizer(max_features=5000)`, `LogisticRegression`, `joblib`, локальный CSV-датасет.
- Интерфейс: Streamlit, `requests`, `pandas`, `altair`.
- Данные: PostgreSQL 16 с постоянной историей предсказаний и опциональными записями об обучении модели.
- Инфраструктура: Nginx, Docker Compose, изолированные сети, именованные тома, проверки состояния контейнеров.
- Мониторинг: сбор `/metrics` через Prometheus и заранее настроенные источник данных и дашборд в Grafana.


## Быстрый старт

1. Создайте локальный файл окружения:

```bash
cp .env.example .env
```

2. Поднимите весь стек:

```bash
docker compose up --build -d
```

3. Проверьте, что все сервисы перешли в рабочее состояние:

```bash
docker compose ps
```

4. Откройте интерфейс и, при необходимости, дашборд мониторинга:

- Интерфейс: `http://localhost/`
- Grafana (через подмаршрут Nginx): `http://localhost/grafana/`

5. При необходимости - логи бэкенда:

```bash
docker compose logs backend
```

Важные значения по умолчанию из `.env.example`:

- `MAX_TEXT_LENGTH=2000`
- `MAX_BATCH_SIZE=32`
- `MODEL_PATH=/app/models/feedback_classifier.joblib`
- `MODEL_METRICS_PATH=/app/models/metrics.json`
- `TRAIN_DATA_PATH=/app/data/train.csv`
- `DATABASE_URL=postgresql+psycopg://rufeedback:rufeedback@postgres:5432/rufeedback`

## Примеры API

Эндпоинты проверки состояния:

```bash
curl -s http://localhost/api/v1/health/live
curl -s http://localhost/api/v1/health/ready
curl -s http://localhost/api/v1/health
curl -s http://localhost/api/v1/model/info
```

Одиночное предсказание:

```bash
curl -s -X POST http://localhost/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Доставка опоздала на два дня, поддержка не отвечает"}'
```

Пакетное предсказание:

```bash
curl -s -X POST http://localhost/api/v1/batch-analyze \
  -H "Content-Type: application/json" \
  -d '{"texts":["Спасибо, все отлично!","Когда будет доставка?","Приложение не работает после обновления"]}'
```

История и агрегированная статистика:

```bash
curl -s "http://localhost/api/v1/predictions?limit=20&offset=0"
curl -s "http://localhost/api/v1/predictions?limit=20&offset=0&label=complaint"
curl -s http://localhost/api/v1/stats
```

Интерфейс использует этот же REST-интерфейс для одиночного предсказания, пакетного предсказания, статуса здоровья на боковой панели, информации о модели, истории и статистики.

## Обучение модели и артефакты

Датасет для обучения в `backend/data/train.csv` и содержит 81 строку вместе с заголовком (минимальное требование по объему данных для четырех меток). При первом запуске бэкенд проверяет, существуют ли уже обученная модель и метрики в томе `model_data`:

- `/app/models/feedback_classifier.joblib`
- `/app/models/metrics.json`

Если любой из этих артефактов отсутствует, бэкенд автоматически обучает модель до того, как перейдет в состояние готовности. При необходимости можно запустить переобучение вручную внутри контейнера backend:

```bash
docker compose exec backend python -m app.ml.train
```

В процессе обучения записываются:

- сериализованный `joblib`-пайплайн;
- `accuracy`;
- `macro_f1`;
- полный `classification_report`;
- размеры train/test выборок;
- имя модели, версия и время обучения.

## База данных и миграции

История предсказаний хранится в PostgreSQL и отображается через SQLAlchemy ORM-модели:

- `prediction_records`
- `model_training_runs`

Бэкенд применяет миграции Alembic при запуске через `alembic upgrade head`, поэтому для проверки достаточно обычного `docker compose up --build -d`. При необходимости ту же команду можно выполнить вручную:

```bash
docker compose exec backend alembic upgrade head
```

Постоянство данных гарантирует, что история предсказаний переживает рестарты бэкенда, потому что состояние хранится в томе `postgres_data`, а не в памяти процесса FastAPI.

## Проверки здоровья и мониторинг

Проверяемые артефакты состояния и мониторинга доступны как через REST-эндпоинты, так и через автоматизацию Compose:

- Проверка состояния API: `GET /api/v1/health/live`, `GET /api/v1/health/ready`, `GET /api/v1/health`
- Эндпоинт метрик: `GET /metrics`
- Проверки состояния сервисов Compose: PostgreSQL, backend, UI, Prometheus, Grafana и Nginx
- Цель Prometheus для сбора метрик: `backend:8000/metrics`
- Grafana: заранее настроенный источник данных Prometheus и дашборд RuFeedback по адресу `/grafana/`

## Выполненные задания

Таблица ниже показывает где реализованы обязательные пункты (8 тем) задания:

| Тема | Что реализовано | Где реализовано |
|---|---|---|
| 1. API-бэкенд | Запуск и остановка FastAPI через lifespan, типизированные Pydantic-контракты запросов и ответов, границы валидации и JSON-обработка ошибок. | `backend/app/main.py`, `backend/app/schemas/feedback.py`, `backend/app/core/errors.py`, `backend/app/api/routes.py` |
| 2. ML-сервис | Изолированный `FeedbackClassifier`, ограниченное потребление ресурсов, локальный пайплайн обучения, сохранение метрик и логирование этапов обработки запроса. | `backend/app/ml/model.py`, `backend/app/ml/train.py`, `backend/app/core/logging.py`, `backend/tests/unit/test_model_lifecycle.py` |
| 3. Интерфейс | Потоки Streamlit для одиночного и пакетного анализа, истории и статистики, интеграция только по REST, визуальные графики и дружелюбная обработка ошибок. | `ui/app.py` |
| 4. Реверс-прокси | Единая точка входа на порту 80, маршрутизация `/api/` в FastAPI, `/` в Streamlit, проксирование `/grafana/` и ограничение частоты API-запросов. | `nginx/nginx.conf` |
| 5. Работа с данными и состоянием | SQLAlchemy ORM-модели, слой репозиториев, хранение в PostgreSQL, JSON-вероятности и версионирование данных через Alembic. | `backend/app/db/models.py`, `backend/app/db/repositories.py`, `backend/alembic/versions/0001_create_prediction_records.py` |
| 6. Оркестрация Docker Compose | Отдельные Dockerfile, файлы зависимостей, сборки с учетом слоев, изолированные сети, именованные тома и порядок запуска с учетом зависимостей. | `backend/Dockerfile`, `ui/Dockerfile`, `backend/requirements.txt`, `ui/requirements.txt`, `docker-compose.yml`, `.dockerignore` |
| 7. Отказоустойчивость | API без хранения состояния в памяти, конфигурация через переменные окружения, корректное завершение работы, сохранность данных при рестарте и `stop_grace_period` для сервисов. | `backend/app/main.py`, `backend/app/core/config.py`, `docker-compose.yml` |
| 8. Проверки здоровья и мониторинг | Эндпоинты живости, готовности и агрегированного состояния, `/metrics`, проверки состояния сервисов Compose, сбор метрик Prometheus, настройка Grafana и дымовая проверка. | `backend/app/api/routes.py`, `backend/app/main.py`, `monitoring/prometheus.yml`, `monitoring/grafana/provisioning/datasources/prometheus.yml`, `monitoring/grafana/provisioning/dashboards/dashboard.yml`, `scripts/smoke_compose.sh` |

## Выбранные доп пункты

Выбранные дополнительные пункты, реализованные в проекте

| Дополнительный пункт | Цель | Подтверждение |
|---|---|---|
| 1 | Своя модель (+5) | `backend/data/train.csv`, `backend/app/ml/train.py`, `backend/app/ml/model.py`, `/app/models/feedback_classifier.joblib`, `/app/models/metrics.json` |
| 2 | Визуальная репрезентация (+5) | Столбчатый график вероятностей для одиночного предсказания, график распределения классов в batch-режиме и визуальные блоки истории/статистики в `ui/app.py` |
| 3 | Метрики Prometheus + Grafana (+3) | `/metrics`, `monitoring/prometheus.yml`, заранее настроенный источник данных Grafana и JSON-дашборд RuFeedback |
| 4 | Кэширование слоев Dockerfile (+1) | `backend/Dockerfile` и `ui/Dockerfile` копируют `requirements.txt` до исходного кода и объясняют цель кэширования комментарием `# Оптимизация сборки` |
| 5 | Менеджмент зависимостей приложения (+1) | `backend/requirements.txt` и `ui/requirements.txt` явно фиксируют зависимости времени выполнения и тестовые зависимости в воспроизводимом виде |

