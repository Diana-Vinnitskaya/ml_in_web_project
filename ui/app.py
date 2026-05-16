from __future__ import annotations

import html
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("UI_API_TIMEOUT", "15"))
LABEL_TITLES = {
    "complaint": "Жалоба",
    "question": "Вопрос",
    "praise": "Похвала",
    "other": "Прочее",
}
LABEL_DESCRIPTIONS = {
    "complaint": "Негативный сигнал: сбой, проблема или неудовлетворённость.",
    "question": "Запрос статуса, деталей или дополнительной информации.",
    "praise": "Позитивный отзыв, благодарность или подтверждение качества.",
    "other": "Нейтральный или смешанный текст вне трёх основных сценариев.",
}
PALETTE = {
    "cerulean": "#4281A4",
    "charcoal": "#343633",
    "linen": "#E0E2DB",
    "carrot": "#FB5012",
}
LABEL_COLOR_BY_KEY = {
    "complaint": PALETTE["carrot"],
    "question": PALETTE["cerulean"],
    "praise": PALETTE["linen"],
    "other": PALETTE["charcoal"],
}
LIGHT_LABEL_KEYS = {"praise"}
EXPECTED_RESPONSE_FIELDS = {
    "id",
    "text",
    "label",
    "confidence",
    "probabilities",
    "processing_time_ms",
    "created_at",
}
BATCH_RESPONSE_FIELDS = {"items", "processing_time_ms"}
HEALTH_RESPONSE_FIELDS = {
    "status",
    "model_loaded",
    "database_available",
    "model_name",
    "detail",
}
MODEL_INFO_FIELDS = {
    "model_name",
    "version",
    "labels",
    "max_text_length",
    "max_batch_size",
    "loaded",
    "metrics",
}
HISTORY_RESPONSE_FIELDS = {"items", "limit", "offset", "total"}
HISTORY_ITEM_FIELDS = {"id", "text", "label", "confidence", "created_at"}
DETAIL_RESPONSE_FIELDS = EXPECTED_RESPONSE_FIELDS | {"model_name", "model_version"}
STATS_RESPONSE_FIELDS = {
    "total_predictions",
    "count_by_label",
    "average_confidence",
    "average_processing_time_ms",
    "last_prediction_at",
}


def get_api_base_url() -> str:
    configured_url = os.getenv("API_BASE_URL")
    if configured_url:
        return configured_url.rstrip("/")

    host = os.getenv("BACKEND_HOST", "localhost").strip() or "localhost"
    port = os.getenv("BACKEND_PORT", "8000").strip() or "8000"
    api_prefix = os.getenv("API_PREFIX", "/api/v1").strip() or "/api/v1"
    if not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"
    if api_prefix != "/":
        api_prefix = api_prefix.rstrip("/")
    return f"http://{host}:{port}{api_prefix}"


def validate_single_text(text: str) -> str | None:
    stripped_text = text.strip()
    if not stripped_text:
        return "Введите текст отзыва перед запуском анализа."
    if len(stripped_text) < MIN_TEXT_LENGTH:
        return f"Введите минимум {MIN_TEXT_LENGTH} символа текста."
    if len(stripped_text) > MAX_TEXT_LENGTH:
        return f"Текст слишком длинный. Максимум {MAX_TEXT_LENGTH} символов."
    return None


def parse_batch_lines(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def validate_batch_text(raw_text: str) -> tuple[list[str] | None, str | None]:
    texts = parse_batch_lines(raw_text)
    if not texts:
        return None, "Добавьте хотя бы одну непустую строку для batch-анализа."
    if len(texts) > MAX_BATCH_SIZE:
        return None, f"Можно отправить не более {MAX_BATCH_SIZE} строк за один раз."

    for index, text in enumerate(texts, start=1):
        if len(text) < MIN_TEXT_LENGTH:
            return (
                None,
                f"Строка {index} слишком короткая. Нужно минимум {MIN_TEXT_LENGTH} символа.",
            )
        if len(text) > MAX_TEXT_LENGTH:
            return (
                None,
                f"Строка {index} слишком длинная. Максимум {MAX_TEXT_LENGTH} символов.",
            )
    return texts, None


def flatten_error_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                location = ".".join(
                    str(part)
                    for part in item.get("loc", [])
                    if part != "body"
                )
                message = str(item.get("msg", "Некорректные данные"))
                messages.append(f"{location}: {message}" if location else message)
            else:
                messages.append(str(item))
        return messages[0] if messages else "Некорректные данные запроса."
    if isinstance(detail, dict):
        return str(detail.get("detail") or detail)
    return "Backend вернул ошибку в неожиданном формате."


def format_validation_error(detail: Any) -> str:
    raw_message = flatten_error_detail(detail)
    normalized = raw_message.lower()
    if "at least" in normalized and "character" in normalized:
        return f"Введите минимум {MIN_TEXT_LENGTH} символа текста."
    if "at most" in normalized and "character" in normalized:
        return f"Текст слишком длинный. Максимум {MAX_TEXT_LENGTH} символов."
    if "field required" in normalized:
        return "Поле с текстом обязательно для анализа."
    return f"Проверьте текст запроса: {raw_message}"


def format_batch_validation_error(detail: Any) -> str:
    raw_message = flatten_error_detail(detail)
    normalized = raw_message.lower()
    if "at least" in normalized and "item" in normalized:
        return "Добавьте хотя бы одну непустую строку для batch-анализа."
    if "at most" in normalized and "item" in normalized:
        return f"Можно отправить не более {MAX_BATCH_SIZE} строк за один раз."
    if "at least" in normalized and "character" in normalized:
        return f"Каждая строка должна содержать минимум {MIN_TEXT_LENGTH} символа."
    if "at most" in normalized and "character" in normalized:
        return f"Одна из строк слишком длинная. Максимум {MAX_TEXT_LENGTH} символов."
    if "field required" in normalized:
        return "Поле texts обязательно для batch-анализа."
    return f"Проверьте строки batch-запроса: {raw_message}"


def format_response_error(status_code: int, detail: Any) -> str:
    if status_code == 422:
        return format_validation_error(detail)
    if status_code == 429:
        return "Слишком много запросов подряд. Подождите пару секунд и попробуйте снова."
    if status_code == 503:
        return "Сервис ещё не готов: модель или база данных временно недоступны."
    if 400 <= status_code < 500:
        return f"Запрос отклонён ({status_code}): {flatten_error_detail(detail)}"
    return (
        f"Во время анализа произошла внутренняя ошибка сервиса ({status_code}). "
        "Попробуйте повторить запрос чуть позже."
    )


def format_batch_response_error(status_code: int, detail: Any) -> str:
    if status_code == 422:
        return format_batch_validation_error(detail)
    if status_code == 429:
        return "Слишком много batch-запросов подряд. Подождите пару секунд и попробуйте снова."
    if status_code == 503:
        return "Batch-анализ пока недоступен: модель или база данных ещё не готовы."
    if 400 <= status_code < 500:
        return f"Batch-запрос отклонён ({status_code}): {flatten_error_detail(detail)}"
    return (
        f"Во время batch-анализа произошла внутренняя ошибка сервиса ({status_code}). "
        "Попробуйте повторить запрос чуть позже."
    )


def request_single_prediction(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        # Слабая связность: UI общается с backend только через REST API.
        response = requests.post(
            f"{get_api_base_url()}/analyze",
            json={"text": text},
            timeout=(3.05, REQUEST_TIMEOUT_SECONDS),
        )
    except requests.Timeout:
        return None, "Backend не ответил вовремя. Попробуйте ещё раз через несколько секунд."
    except requests.RequestException:
        return None, "Не удалось связаться с backend. Проверьте, что сервис запущен."

    try:
        payload = response.json()
    except ValueError:
        return None, "Backend вернул некорректный JSON. Проверьте состояние сервиса."

    if response.status_code != 200:
        return None, format_response_error(response.status_code, payload.get("detail"))

    if not EXPECTED_RESPONSE_FIELDS.issubset(payload):
        return None, "Backend вернул неполный ответ. Проверьте контракт сервиса."

    return payload, None


def request_batch_prediction(
    texts: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(
            f"{get_api_base_url()}/batch-analyze",
            json={"texts": texts},
            timeout=(3.05, REQUEST_TIMEOUT_SECONDS),
        )
    except requests.Timeout:
        return None, "Batch backend не ответил вовремя. Попробуйте ещё раз через несколько секунд."
    except requests.RequestException:
        return None, "Не удалось связаться с backend для batch-анализа. Проверьте, что сервис запущен."

    try:
        payload = response.json()
    except ValueError:
        return None, "Backend вернул некорректный JSON для batch-анализа. Проверьте состояние сервиса."

    if response.status_code != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        return None, format_batch_response_error(response.status_code, detail)

    if not isinstance(payload, dict) or not BATCH_RESPONSE_FIELDS.issubset(payload):
        return None, "Backend вернул неполный batch-ответ. Проверьте контракт сервиса."
    if not isinstance(payload.get("items"), list):
        return None, "Backend вернул batch-ответ в неожиданном формате."
    if any(not EXPECTED_RESPONSE_FIELDS.issubset(item) for item in payload["items"]):
        return None, "Один из элементов batch-ответа не соответствует контракту."

    return payload, None


def request_get_json(
    path: str,
    *,
    params: dict[str, Any] | None,
    timeout_message: str,
    request_error_message: str,
    invalid_json_message: str,
) -> tuple[Any | None, int | None, str | None]:
    try:
        response = requests.get(
            f"{get_api_base_url()}{path}",
            params=params,
            timeout=(3.05, REQUEST_TIMEOUT_SECONDS),
        )
    except requests.Timeout:
        return None, None, timeout_message
    except requests.RequestException:
        return None, None, request_error_message

    try:
        payload = response.json()
    except ValueError:
        return None, response.status_code, invalid_json_message

    return payload, response.status_code, None


def request_service_health() -> tuple[dict[str, Any] | None, str | None]:
    payload, status_code, transport_error = request_get_json(
        "/health",
        params=None,
        timeout_message="Проверка health backend заняла слишком много времени.",
        request_error_message="Не удалось получить health backend. Проверьте, что сервис запущен.",
        invalid_json_message="Backend вернул некорректный JSON для health-check.",
    )
    if transport_error:
        return None, transport_error
    if status_code not in {200, 503}:
        detail = flatten_error_detail(payload.get("detail")) if isinstance(payload, dict) else payload
        return None, f"Health-check backend завершился ошибкой ({status_code}): {detail}"
    if not isinstance(payload, dict) or not HEALTH_RESPONSE_FIELDS.issubset(payload):
        return None, "Health-check backend вернул неполный ответ."
    return payload, None


def request_model_info() -> tuple[dict[str, Any] | None, str | None]:
    payload, status_code, transport_error = request_get_json(
        "/model/info",
        params=None,
        timeout_message="Профиль модели загружается слишком долго.",
        request_error_message="Не удалось получить профиль модели от backend.",
        invalid_json_message="Backend вернул некорректный JSON для профиля модели.",
    )
    if transport_error:
        return None, transport_error
    if status_code != 200:
        detail = flatten_error_detail(payload.get("detail")) if isinstance(payload, dict) else payload
        if status_code == 503:
            return None, f"Профиль модели пока недоступен: {detail}"
        return None, f"Не удалось получить профиль модели ({status_code}): {detail}"
    if not isinstance(payload, dict) or not MODEL_INFO_FIELDS.issubset(payload):
        return None, "Backend вернул неполный профиль модели."
    return payload, None


def request_prediction_history(
    *,
    label: str | None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[dict[str, Any] | None, str | None]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if label:
        params["label"] = label

    payload, status_code, transport_error = request_get_json(
        "/predictions",
        params=params,
        timeout_message="История предсказаний загружается слишком долго.",
        request_error_message="Не удалось получить историю предсказаний от backend.",
        invalid_json_message="Backend вернул некорректный JSON для истории предсказаний.",
    )
    if transport_error:
        return None, transport_error
    if status_code != 200:
        detail = flatten_error_detail(payload.get("detail")) if isinstance(payload, dict) else payload
        if status_code == 503:
            return None, f"История пока недоступна: {detail}"
        return None, f"Не удалось получить историю предсказаний ({status_code}): {detail}"
    if not isinstance(payload, dict) or not HISTORY_RESPONSE_FIELDS.issubset(payload):
        return None, "Backend вернул неполный ответ для истории предсказаний."
    items = payload.get("items")
    if not isinstance(items, list):
        return None, "Backend вернул историю предсказаний в неожиданном формате."
    if any(not isinstance(item, dict) or not HISTORY_ITEM_FIELDS.issubset(item) for item in items):
        return None, "Одна из строк истории не соответствует REST-контракту."
    return payload, None


def request_prediction_detail(
    prediction_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    payload, status_code, transport_error = request_get_json(
        f"/predictions/{prediction_id}",
        params=None,
        timeout_message="Детали предсказания загружаются слишком долго.",
        request_error_message="Не удалось получить детали предсказания от backend.",
        invalid_json_message="Backend вернул некорректный JSON для деталей предсказания.",
    )
    if transport_error:
        return None, transport_error
    if status_code != 200:
        detail = flatten_error_detail(payload.get("detail")) if isinstance(payload, dict) else payload
        if status_code == 404:
            return None, "Запись из истории не найдена. Возможно, база была пересоздана."
        if status_code == 503:
            return None, f"Детали предсказания пока недоступны: {detail}"
        return None, f"Не удалось получить детали предсказания ({status_code}): {detail}"
    if not isinstance(payload, dict) or not DETAIL_RESPONSE_FIELDS.issubset(payload):
        return None, "Backend вернул неполные детали предсказания."
    return payload, None


def request_prediction_stats() -> tuple[dict[str, Any] | None, str | None]:
    payload, status_code, transport_error = request_get_json(
        "/stats",
        params=None,
        timeout_message="Статистика загружается слишком долго.",
        request_error_message="Не удалось получить статистику от backend.",
        invalid_json_message="Backend вернул некорректный JSON для статистики.",
    )
    if transport_error:
        return None, transport_error
    if status_code != 200:
        detail = flatten_error_detail(payload.get("detail")) if isinstance(payload, dict) else payload
        if status_code == 503:
            return None, f"Статистика пока недоступна: {detail}"
        return None, f"Не удалось получить статистику ({status_code}): {detail}"
    if not isinstance(payload, dict) or not STATS_RESPONSE_FIELDS.issubset(payload):
        return None, "Backend вернул неполный ответ для статистики."
    count_by_label = payload.get("count_by_label")
    if not isinstance(count_by_label, dict) or any(label not in count_by_label for label in LABEL_TITLES):
        return None, "Backend вернул статистику по классам в неожиданном формате."
    return payload, None


def format_timestamp(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return str(value).replace("T", " ")


def shorten_text(text: str, *, limit: int = 90) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def format_label_title(label: str) -> str:
    return LABEL_TITLES.get(label, label)


def render_html_block(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def inject_theme_overrides() -> None:
    st.markdown(
        f"""
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap");

        :root {{
            --cerulean: {PALETTE["cerulean"]};
            --charcoal: {PALETTE["charcoal"]};
            --linen: {PALETTE["linen"]};
            --carrot: {PALETTE["carrot"]};
            --paper: #F7F3EC;
            --paper-strong: rgba(255, 255, 255, 0.78);
            --paper-soft: rgba(255, 255, 255, 0.58);
            --line: rgba(52, 54, 51, 0.11);
            --shadow: rgba(52, 54, 51, 0.10);
            --copy-soft: rgba(52, 54, 51, 0.74);
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(251, 80, 18, 0.17) 0%, transparent 28%),
                radial-gradient(circle at 100% 8%, rgba(66, 129, 164, 0.20) 0%, transparent 32%),
                linear-gradient(180deg, #F6F3EC 0%, #EEF0EB 38%, #F5F4EF 100%);
            color: var(--charcoal);
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp input,
        .stApp textarea,
        .stApp button,
        .stApp [data-testid="stMarkdownContainer"] {{
            font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
        }}

        h1,
        h2,
        h3,
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .hero-title {{
            font-family: "Fraunces", Georgia, serif !important;
            letter-spacing: -0.03em;
        }}

        h1,
        .stMarkdown h1 {{
            font-size: clamp(2.4rem, 4vw, 4rem);
            line-height: 0.98;
            margin-bottom: 0.5rem;
        }}

        h2,
        .stMarkdown h2 {{
            font-size: clamp(1.65rem, 2.6vw, 2.45rem);
        }}

        h3,
        .stMarkdown h3 {{
            font-size: clamp(1.55rem, 3vw, 2.35rem) !important;
            line-height: 1.05;
            margin-bottom: 0.55rem;
        }}

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stAppDeployButton,
        #MainMenu,
        footer {{
            display: none !important;
            visibility: hidden !important;
        }}

        .stMainBlockContainer {{
            padding-top: 0 !important;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(52, 54, 51, 0.98) 0%, #2F312F 100%);
            border-right: 1px solid rgba(224, 226, 219, 0.15);
        }}

        section[data-testid="stSidebar"] * {{
            color: #F4F1EA;
        }}

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: rgba(244, 241, 234, 0.72);
        }}

        section[data-testid="stSidebar"] div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(224, 226, 219, 0.14);
            box-shadow: none;
        }}

        section[data-testid="stSidebar"] code {{
            color: #FFD8C9 !important;
            background: rgba(251, 80, 18, 0.16);
        }}

        div[data-testid="stMetric"] {{
            background: var(--paper-strong);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-shadow: 0 14px 30px var(--shadow);
        }}

        div[data-testid="stMetric"] label {{
            font-size: 0.80rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(52, 54, 51, 0.66);
        }}

        div[data-testid="stMetricValue"] {{
            font-family: "Fraunces", Georgia, serif;
            color: var(--charcoal);
        }}

        div[data-testid="stMetricValue"] p,
        div[data-testid="stMetricLabel"] p {{
            overflow-wrap: anywhere;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 28px;
            background: var(--paper-strong);
            border: 1px solid var(--line);
            box-shadow: 0 18px 42px var(--shadow);
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            background: linear-gradient(135deg, var(--carrot) 0%, #FF7440 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 999px;
            min-height: 3rem;
            font-weight: 700;
            padding-inline: 1.15rem;
            box-shadow: 0 14px 28px rgba(251, 80, 18, 0.22);
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            border: none;
            color: #FFFFFF;
            transform: translateY(-1px);
        }}

        .stButton > button:focus,
        .stFormSubmitButton > button:focus {{
            box-shadow: 0 0 0 0.2rem rgba(251, 80, 18, 0.20) !important;
        }}

        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        textarea {{
            border-radius: 18px !important;
            border: 1px solid var(--line) !important;
            background: rgba(255, 255, 255, 0.92) !important;
        }}

        div[data-baseweb="select"] > div {{
            min-height: 3rem;
        }}

        div[data-baseweb="base-input"] input,
        div[data-baseweb="select"] input,
        textarea {{
            color: var(--charcoal) !important;
            -webkit-text-fill-color: var(--charcoal) !important;
            caret-color: var(--charcoal) !important;
            line-height: 1.58;
        }}

        textarea::placeholder,
        div[data-baseweb="base-input"] input::placeholder {{
            color: rgba(52, 54, 51, 0.52) !important;
            -webkit-text-fill-color: rgba(52, 54, 51, 0.52) !important;
            opacity: 1 !important;
        }}

        [data-testid="InputInstructions"] {{
            color: rgba(52, 54, 51, 0.68) !important;
        }}

        [data-testid="InputInstructions"] * {{
            color: inherit !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.35rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: auto;
            border-radius: 999px;
            padding: 0.72rem 1rem;
            color: rgba(52, 54, 51, 0.72);
            font-weight: 700;
            justify-content: center;
            width: 100%;
            min-width: 0;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--charcoal) !important;
            color: #FFFFFF !important;
        }}

        .stTabs [data-baseweb="tab-highlight"] {{
            display: none;
        }}

        [data-testid="stDataFrame"] {{
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid var(--line);
        }}

        .hero-shell {{
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.84) 0%, rgba(255, 255, 255, 0.58) 100%);
            border: 1px solid var(--line);
            border-radius: 32px;
            box-shadow: 0 24px 48px var(--shadow);
            padding: 1.6rem 1.7rem;
            margin-bottom: 1.4rem;
            overflow: hidden;
            position: relative;
        }}

        .hero-shell::after {{
            content: "";
            position: absolute;
            inset: auto -60px -90px auto;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(66, 129, 164, 0.24) 0%, rgba(66, 129, 164, 0) 72%);
            pointer-events: none;
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.95fr);
            gap: 1rem;
            align-items: stretch;
        }}

        .eyebrow {{
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--cerulean);
            margin-bottom: 0.6rem;
        }}

        .hero-copy {{
            max-width: 39rem;
            font-size: 1.03rem;
            line-height: 1.65;
            color: var(--copy-soft);
            margin: 0;
        }}

        .hero-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.42rem 0.76rem;
            background: rgba(66, 129, 164, 0.10);
            border: 1px solid rgba(66, 129, 164, 0.14);
            font-size: 0.81rem;
            font-weight: 700;
        }}

        .hero-badge--warm {{
            background: rgba(251, 80, 18, 0.10);
            border-color: rgba(251, 80, 18, 0.14);
        }}

        .hero-panel {{
            background: linear-gradient(145deg, rgba(52, 54, 51, 0.98) 0%, rgba(66, 129, 164, 0.94) 100%);
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1rem;
            color: #FFFFFF;
            position: relative;
            z-index: 1;
        }}

        .hero-panel * {{
            color: #FFFFFF;
        }}

        .hero-panel-title {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            opacity: 0.78;
            margin-bottom: 0.75rem;
            font-weight: 800;
        }}

        .hero-kpis {{
            display: grid;
            gap: 0.7rem;
        }}

        .hero-kpi {{
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: baseline;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.14);
        }}

        .hero-kpi:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}

        .hero-kpi span {{
            font-size: 0.88rem;
            opacity: 0.78;
        }}

        .hero-kpi strong {{
            font-size: 1rem;
            font-weight: 800;
            text-align: right;
        }}

        .hero-panel-note {{
            margin-top: 0.85rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.84rem;
            line-height: 1.45;
            color: rgba(255, 255, 255, 0.78);
        }}

        .section-kicker {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 800;
            color: var(--cerulean);
            margin: 0 0 0.45rem;
        }}

        .panel-copy {{
            color: var(--copy-soft);
        }}

        .result-banner {{
            border-radius: 22px;
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(52, 54, 51, 0.08);
        }}

        .result-banner--success {{
            background: rgba(251, 80, 18, 0.10);
        }}

        .result-banner--info {{
            background: rgba(66, 129, 164, 0.11);
        }}

        .result-banner__title {{
            display: block;
            margin-bottom: 0.25rem;
            font-weight: 800;
        }}

        .result-banner p {{
            margin: 0;
            color: var(--copy-soft);
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.8rem;
            margin-bottom: 1rem;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 26px rgba(52, 54, 51, 0.06);
            min-width: 0;
        }}

        .stat-label {{
            display: block;
            margin-bottom: 0.5rem;
            color: rgba(52, 54, 51, 0.58);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        .stat-value {{
            display: block;
            color: var(--charcoal);
            font-family: "Fraunces", Georgia, serif;
            font-size: clamp(1.8rem, 5vw, 3rem);
            line-height: 0.95;
            overflow-wrap: anywhere;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.55rem;
        }}

        .meta-chip {{
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.8rem 0.95rem;
        }}

        .meta-chip strong {{
            display: block;
            margin-bottom: 0.28rem;
            font-size: 0.82rem;
            color: rgba(52, 54, 51, 0.62);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .meta-chip code,
        .meta-chip span {{
            overflow-wrap: anywhere;
            word-break: break-word;
        }}

        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin-top: 0.3rem;
        }}

        .legend-card {{
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.9rem;
        }}

        .legend-head {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .legend-dot {{
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 999px;
            flex: 0 0 auto;
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.72);
        }}

        .legend-title {{
            font-weight: 800;
            color: var(--charcoal);
        }}

        .legend-key {{
            display: block;
            color: rgba(52, 54, 51, 0.52);
            font-size: 0.8rem;
            margin-top: 0.08rem;
        }}

        .legend-copy {{
            margin-top: 0.55rem;
            color: var(--copy-soft);
            font-size: 0.91rem;
            line-height: 1.5;
        }}

        .metric-bar-group {{
            display: grid;
            gap: 0.8rem;
            margin-top: 0.35rem;
        }}

        .metric-bar-row {{
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.82rem 0.95rem;
            box-shadow: 0 10px 22px rgba(52, 54, 51, 0.05);
        }}

        .metric-bar-meta {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }}

        .metric-bar-label {{
            font-weight: 700;
            color: var(--charcoal);
        }}

        .metric-bar-value {{
            color: rgba(52, 54, 51, 0.7);
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }}

        .metric-bar-track {{
            height: 0.78rem;
            background: rgba(52, 54, 51, 0.09);
            border-radius: 999px;
            overflow: hidden;
        }}

        .metric-bar-fill {{
            height: 100%;
            border-radius: 999px;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.20);
        }}

        .metric-bar-fill--light {{
            border: 1px solid rgba(52, 54, 51, 0.14);
            box-sizing: border-box;
        }}

        @media (max-width: 980px) {{
            .block-container {{
                padding-top: 0.8rem;
                padding-bottom: 3rem;
            }}

            .hero-shell {{
                padding: 1.15rem;
            }}

            .hero-grid {{
                grid-template-columns: 1fr;
            }}

            .hero-panel {{
                padding: 1rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                padding: 0.68rem 0.45rem;
                font-size: 0.95rem;
            }}

            div[data-testid="stMetric"] {{
                padding: 0.85rem 0.9rem;
            }}

            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] p {{
                font-size: clamp(1.75rem, 8vw, 2.5rem) !important;
                line-height: 0.98 !important;
            }}
        }}

        @media (max-width: 720px) {{
            .hero-title {{
                font-size: clamp(2rem, 10vw, 2.7rem) !important;
            }}

            .section-kicker {{
                font-size: 0.72rem;
            }}

            h3,
            .stMarkdown h3 {{
                font-size: clamp(1.35rem, 8vw, 1.9rem) !important;
            }}

            .hero-badges {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}

            .hero-badge {{
                justify-content: center;
                text-align: center;
            }}

            .stat-grid {{
                grid-template-columns: 1fr;
            }}

            .meta-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_banner(title: str, description: str, *, tone: str) -> None:
    render_html_block(
        "".join(
            [
                f'<div class="result-banner result-banner--{html.escape(tone)}">',
                f'<span class="result-banner__title">{html.escape(title)}</span>',
                f"<p>{html.escape(description)}</p>",
                "</div>",
            ]
        )
    )


def render_stat_cards(cards: list[dict[str, str]]) -> None:
    html_cards: list[str] = []
    for card in cards:
        html_cards.append(
            "".join(
                [
                    '<div class="stat-card">',
                    f'<span class="stat-label">{html.escape(card["label"])}</span>',
                    f'<strong class="stat-value">{html.escape(card["value"])}</strong>',
                    "</div>",
                ]
            )
        )

    render_html_block('<div class="stat-grid">' + "".join(html_cards) + "</div>")


def render_meta_chips(chips: list[dict[str, str]]) -> None:
    html_chips: list[str] = []
    for chip in chips:
        value_is_code = chip.get("code", "false") == "true"
        value_markup = (
            f'<code>{html.escape(chip["value"])}</code>'
            if value_is_code
            else f'<span>{html.escape(chip["value"])}</span>'
        )
        html_chips.append(
            "".join(
                [
                    '<div class="meta-chip">',
                    f'<strong>{html.escape(chip["label"])}</strong>',
                    value_markup,
                    "</div>",
                ]
            )
        )

    render_html_block('<div class="meta-grid">' + "".join(html_chips) + "</div>")


def render_page_intro(
    health_payload: dict[str, Any] | None,
    model_payload: dict[str, Any] | None,
    health_error: str | None,
) -> None:
    if health_payload is None:
        status_title = "Связь с backend не подтверждена"
    elif health_payload.get("status") == "ok":
        status_title = "Сервис готов к работе"
    else:
        status_title = "Backend отвечает, но не полностью готов"

    model_name = "Модель ещё не ответила"
    if model_payload and model_payload.get("model_name"):
        model_name = str(model_payload["model_name"])

    model_version = "n/a"
    if model_payload and model_payload.get("version"):
        model_version = str(model_payload["version"])

    model_loaded = "n/a"
    database_available = "n/a"
    if health_payload is not None:
        model_loaded = "Да" if bool(health_payload.get("model_loaded")) else "Нет"
        database_available = (
            "Да" if bool(health_payload.get("database_available")) else "Нет"
        )

    status_note = ""
    if health_error:
        status_note = health_error
    elif health_payload and health_payload.get("detail"):
        status_note = str(health_payload["detail"])

    hero_markup = f"""
    <section class="hero-shell">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">Local ML workspace</div>
          <div class="hero-title">RuFeedback Classifier</div>
          <div class="hero-badges">
            <span class="hero-badge">REST UI</span>
            <span class="hero-badge hero-badge--warm">4 класса</span>
          </div>
        </div>
        <div class="hero-panel">
          <div class="hero-panel-title">Сервис сейчас</div>
          <div class="hero-kpis">
            <div class="hero-kpi">
              <span>Backend</span>
              <strong>{html.escape(status_title)}</strong>
            </div>
            <div class="hero-kpi">
              <span>Модель загружена</span>
              <strong>{html.escape(model_loaded)}</strong>
            </div>
            <div class="hero-kpi">
              <span>База данных</span>
              <strong>{html.escape(database_available)}</strong>
            </div>
            <div class="hero-kpi">
              <span>Модель backend</span>
              <strong>{html.escape(model_name)} v{html.escape(model_version)}</strong>
            </div>
          </div>
          {f'<div class="hero-panel-note">{html.escape(status_note)}</div>' if status_note else ''}
        </div>
      </div>
    </section>
    """
    render_html_block(hero_markup)


def render_label_legend() -> None:
    cards: list[str] = []
    for key, title in LABEL_TITLES.items():
        cards.append(
            "".join(
                [
                    '<div class="legend-card">',
                    '<div class="legend-head">',
                    (
                        f'<span class="legend-dot" style="background: '
                        f'{html.escape(LABEL_COLOR_BY_KEY[key])};"></span>'
                    ),
                    '<div>',
                    f'<span class="legend-title">{html.escape(title)}</span>',
                    f'<span class="legend-key">{html.escape(key)}</span>',
                    "</div>",
                    "</div>",
                    f'<div class="legend-copy">{html.escape(LABEL_DESCRIPTIONS[key])}</div>',
                    "</div>",
                ]
            )
        )

    render_html_block('<div class="legend-grid">' + "".join(cards) + "</div>")


def render_section_heading(kicker: str, title: str, description: str | None = None) -> None:
    st.markdown(
        f'<p class="section-kicker">{html.escape(kicker)}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def render_metric_bars(
    rows: list[dict[str, Any]],
    *,
    scale_max: float,
) -> None:
    safe_scale_max = scale_max if scale_max > 0 else 1.0
    html_rows: list[str] = []

    for row in rows:
        value = max(float(row["value"]), 0.0)
        width_percent = max(min((value / safe_scale_max) * 100, 100), 0)
        label = html.escape(str(row["label"]))
        value_text = html.escape(str(row["value_text"]))
        color = html.escape(str(row["color"]))
        extra_class = " metric-bar-fill--light" if bool(row.get("is_light")) else ""
        html_rows.append(
            "".join(
                [
                    '<div class="metric-bar-row">',
                    '<div class="metric-bar-meta">',
                    f'<span class="metric-bar-label">{label}</span>',
                    f'<span class="metric-bar-value">{value_text}</span>',
                    "</div>",
                    '<div class="metric-bar-track">',
                    (
                        f'<div class="metric-bar-fill{extra_class}" '
                        f'style="width: {width_percent:.1f}%; background: {color};"></div>'
                    ),
                    "</div>",
                    "</div>",
                ]
            )
        )

    html_markup = '<div class="metric-bar-group">' + "".join(html_rows) + "</div>"
    render_html_block(html_markup)


def render_probability_chart(probabilities: dict[str, float]) -> None:
    # Визуальная репрезентация
    rows = [
        {
            "label": LABEL_TITLES.get(key, key),
            "value": float(probabilities.get(key, 0.0)),
            "value_text": f"{float(probabilities.get(key, 0.0)) * 100:.1f}%",
            "color": LABEL_COLOR_BY_KEY.get(key, "#475569"),
            "is_light": key in LIGHT_LABEL_KEYS,
        }
        for key in LABEL_TITLES
    ]
    render_metric_bars(rows, scale_max=1.0)


def make_history_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Создано": format_timestamp(item["created_at"]),
                "Класс": format_label_title(str(item["label"])),
                "Уверенность": float(item["confidence"]),
                "Текст": shorten_text(str(item["text"])),
                "Prediction ID": str(item["id"]),
            }
            for item in items
        ]
    )


def make_batch_results_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "№": index,
                "Текст": item["text"],
                "Класс": LABEL_TITLES.get(str(item["label"]), str(item["label"])),
                "Уверенность": float(item["confidence"]),
                "Prediction ID": item["id"],
                "Создано": str(item["created_at"]).replace("T", " "),
            }
            for index, item in enumerate(items, start=1)
        ]
    )


def make_batch_distribution_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    counts = {
        label: sum(1 for item in items if item["label"] == label)
        for label in LABEL_TITLES
    }
    return pd.DataFrame(
        [
            {
                "key": key,
                "Класс": LABEL_TITLES.get(key, key),
                "Количество": counts[key],
            }
            for key in LABEL_TITLES
        ]
    )


def render_batch_distribution_chart(items: list[dict[str, Any]]) -> None:
    distribution_frame = make_batch_distribution_frame(items)
    rows = [
        {
            "label": str(row["Класс"]),
            "value": float(row["Количество"]),
            "value_text": str(int(row["Количество"])),
            "color": LABEL_COLOR_BY_KEY.get(str(row["key"]), "#475569"),
            "is_light": str(row["key"]) in LIGHT_LABEL_KEYS,
        }
        for row in distribution_frame.to_dict("records")
    ]
    scale_max = max((float(row["value"]) for row in rows), default=1.0)
    render_metric_bars(rows, scale_max=scale_max)


def render_stats_distribution_chart(count_by_label: dict[str, Any]) -> None:
    rows = [
        {
            "label": LABEL_TITLES.get(key, key),
            "value": float(count_by_label.get(key, 0)),
            "value_text": str(int(count_by_label.get(key, 0))),
            "color": LABEL_COLOR_BY_KEY.get(key, "#475569"),
            "is_light": key in LIGHT_LABEL_KEYS,
        }
        for key in LABEL_TITLES
    ]
    scale_max = max((float(row["value"]) for row in rows), default=1.0)
    render_metric_bars(rows, scale_max=scale_max)


def render_prediction_result(result: dict[str, Any]) -> None:
    label = str(result["label"])
    confidence = float(result["confidence"])
    processing_time_ms = float(result["processing_time_ms"])

    with st.container(border=True):
        render_result_banner(
            f"Single-анализ завершён: {format_label_title(label)}",
            "Запись уже сохранена в историю и доступна в reviewer-блоке ниже.",
            tone="success",
        )
        render_stat_cards(
            [
                {"label": "Класс", "value": format_label_title(label)},
                {"label": "Уверенность", "value": f"{confidence * 100:.1f}%"},
                {"label": "Время, ms", "value": f"{processing_time_ms:.1f}"},
            ]
        )
        render_meta_chips(
            [
                {"label": "Prediction ID", "value": str(result["id"]), "code": "true"},
                {"label": "Создано", "value": format_timestamp(result["created_at"])},
            ]
        )
        st.markdown("**Исходный текст**")
        st.write(str(result["text"]))
        st.markdown("**Распределение вероятностей по классам**")
        render_probability_chart(dict(result["probabilities"]))


def render_prediction_detail(detail: dict[str, Any]) -> None:
    label = str(detail["label"])
    with st.container(border=True):
        render_result_banner(
            f"Детали записи: {format_label_title(label)}",
            "Полная карточка выбранного предсказания из reviewer dashboard.",
            tone="info",
        )
        render_stat_cards(
            [
                {"label": "Класс", "value": format_label_title(label)},
                {
                    "label": "Уверенность",
                    "value": f"{float(detail['confidence']) * 100:.1f}%",
                },
                {
                    "label": "Время, ms",
                    "value": f"{float(detail['processing_time_ms']):.1f}",
                },
            ]
        )
        render_meta_chips(
            [
                {"label": "Prediction ID", "value": str(detail["id"]), "code": "true"},
                {
                    "label": "Модель",
                    "value": f'{detail["model_name"]} v{detail["model_version"]}',
                    "code": "true",
                },
            ]
        )
        st.caption(f"Сохранено: {format_timestamp(detail['created_at'])}")
        st.markdown("**Полный текст**")
        st.write(str(detail["text"]))
        st.markdown("**Вероятности по классам**")
        render_probability_chart(dict(detail["probabilities"]))


def render_batch_result(result: dict[str, Any]) -> None:
    items = list(result["items"])
    total_processing_time_ms = float(result["processing_time_ms"])
    average_time_ms = total_processing_time_ms / max(len(items), 1)

    with st.container(border=True):
        render_result_banner(
            f"Batch готов: сохранено {len(items)} записей",
            "Каждая строка обработана отдельной записью истории через REST batch endpoint.",
            tone="success",
        )
        render_stat_cards(
            [
                {"label": "Строк", "value": str(len(items))},
                {"label": "Всего, ms", "value": f"{total_processing_time_ms:.1f}"},
                {"label": "Среднее, ms", "value": f"{average_time_ms:.1f}"},
            ]
        )

        st.markdown("**Результаты по строкам**")
        batch_frame = make_batch_results_frame(items)
        st.dataframe(
            batch_frame.style.format({"Уверенность": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Распределение классов в batch**")
        render_batch_distribution_chart(items)


def render_sidebar_status(
    health_payload: dict[str, Any] | None,
    health_error: str | None,
    model_payload: dict[str, Any] | None,
    model_error: str | None,
) -> None:
    with st.sidebar:
        st.markdown("## Control room")

        with st.container(border=True):
            st.markdown("### Backend status")
            if health_error:
                st.error(health_error)
            elif health_payload is None:
                st.warning("Статус backend пока недоступен.")
            else:
                if health_payload["status"] == "ok":
                    st.success("Backend готов к работе")
                else:
                    st.warning("Backend отвечает, но ещё не полностью готов")
                status_left, status_right = st.columns(2)
                status_left.metric(
                    "Модель загружена",
                    "Да" if bool(health_payload["model_loaded"]) else "Нет",
                )
                status_right.metric(
                    "База данных",
                    "Да" if bool(health_payload["database_available"]) else "Нет",
                )
                if health_payload.get("model_name"):
                    st.caption(f"Текущая модель backend: `{health_payload['model_name']}`")
                if health_payload.get("detail"):
                    st.caption(str(health_payload["detail"]))

        with st.container(border=True):
            st.markdown("### Model profile")
            if model_error:
                st.warning(model_error)
            elif model_payload is None:
                st.info("Профиль модели пока недоступен.")
            else:
                st.markdown(f"**{model_payload['model_name']}**")
                st.caption(f"Версия `{model_payload['version']}`")
                labels = ", ".join(
                    f"`{label}`" for label in list(model_payload.get("labels", []))
                )
                st.markdown(f"**Labels:** {labels}")
                limits_left, limits_right = st.columns(2)
                limits_left.metric("MAX_TEXT_LENGTH", str(model_payload["max_text_length"]))
                limits_right.metric("MAX_BATCH_SIZE", str(model_payload["max_batch_size"]))
                metrics = model_payload.get("metrics") or {}
                metrics_left, metrics_right = st.columns(2)
                accuracy = metrics.get("accuracy")
                macro_f1 = metrics.get("macro_f1")
                metrics_left.metric(
                    "Accuracy",
                    "n/a" if accuracy is None else f"{float(accuracy) * 100:.1f}%",
                )
                metrics_right.metric(
                    "Macro F1",
                    "n/a" if macro_f1 is None else f"{float(macro_f1) * 100:.1f}%",
                )


st.set_page_config(
    page_title="RuFeedback Classifier",
    page_icon=":material/rate_review:",
    layout="wide",
)

if "single_result" not in st.session_state:
    st.session_state["single_result"] = None
if "single_error" not in st.session_state:
    st.session_state["single_error"] = None
if "batch_result" not in st.session_state:
    st.session_state["batch_result"] = None
if "batch_error" not in st.session_state:
    st.session_state["batch_error"] = None

inject_theme_overrides()

health_payload, health_error = request_service_health()
model_payload, model_error = request_model_info()

render_sidebar_status(health_payload, health_error, model_payload, model_error)
render_page_intro(health_payload, model_payload, health_error)

analysis_tab, batch_tab, reviewer_tab = st.tabs(
    ["Single-анализ", "Batch-поток", "Reviewer dashboard"]
)

with analysis_tab:
    render_section_heading(
        "Single flow",
        "Разбор одного отзыва",
    )
    left_column, right_column = st.columns([1.3, 0.85], gap="large")

    with left_column:
        with st.container(border=True):
            st.markdown("#### Вставьте текст отзыва")
            st.caption(
                f"Допустимая длина: от {MIN_TEXT_LENGTH} до {MAX_TEXT_LENGTH} символов."
            )
            with st.form("single_analysis_form", clear_on_submit=False):
                text = st.text_area(
                    "Текст отзыва",
                    height=230,
                    placeholder=(
                        "Например: Доставка опоздала на два дня, поддержка не отвечает"
                    ),
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "Анализировать и сохранить",
                    use_container_width=True,
                    type="primary",
                )

        if submitted:
            validation_error = validate_single_text(text)
            if validation_error:
                st.session_state["single_result"] = None
                st.session_state["single_error"] = validation_error
            else:
                with st.spinner("Анализируем отзыв, считаем вероятности и сохраняем результат..."):
                    result, error = request_single_prediction(text.strip())
                st.session_state["single_result"] = result
                st.session_state["single_error"] = error

        if st.session_state["single_error"]:
            st.error(st.session_state["single_error"])

        if st.session_state["single_result"]:
            render_prediction_result(st.session_state["single_result"])

    with right_column:
        with st.container(border=True):
            st.markdown("#### Что вернётся")
            st.write(
                "Backend возвращает класс, уверенность модели, вероятности по всем "
                "классам, время обработки и идентификатор сохранённой записи."
            )

        with st.container(border=True):
            st.markdown("#### Классы модели")
            render_label_legend()

        with st.container(border=True):
            st.markdown("#### REST endpoint")
            st.code(get_api_base_url(), language="text")

with batch_tab:
    render_section_heading(
        "Batch lane",
        "Обработка группы отзывов",
    )
    batch_left_column, batch_right_column = st.columns([1.25, 0.9], gap="large")

    with batch_left_column:
        with st.container(border=True):
            st.markdown("#### Отзывы по одному на строку")
            st.caption(
                "Пустые строки будут отброшены автоматически до отправки в backend."
            )
            with st.form("batch_analysis_form", clear_on_submit=False):
                batch_text = st.text_area(
                    "Отзывы по одному на строку",
                    height=250,
                    placeholder=(
                        "Спасибо за оперативный ответ\n"
                        "Когда уже доставят заказ?\n"
                        "Приложение вылетает после обновления"
                    ),
                    label_visibility="collapsed",
                )
                st.caption(
                    "Лимит: до "
                    f"{MAX_BATCH_SIZE} отзывов, каждый от {MIN_TEXT_LENGTH} до {MAX_TEXT_LENGTH} символов."
                )
                batch_submitted = st.form_submit_button(
                    "Запустить batch-анализ",
                    use_container_width=True,
                    type="primary",
                )

        if batch_submitted:
            texts, validation_error = validate_batch_text(batch_text)
            if validation_error:
                st.session_state["batch_result"] = None
                st.session_state["batch_error"] = validation_error
            else:
                with st.spinner("Разбиваем строки, отправляем batch в backend и сохраняем результаты..."):
                    progress = st.progress(15, text="Подготавливаем batch-запрос...")
                    progress.progress(45, text=f"Отправляем {len(texts or [])} строк в backend...")
                    result, error = request_batch_prediction(texts or [])
                    progress.progress(100, text="Результаты batch-анализа получены.")
                    progress.empty()
                st.session_state["batch_result"] = result
                st.session_state["batch_error"] = error

        if st.session_state["batch_error"]:
            st.error(st.session_state["batch_error"])

        if st.session_state["batch_result"]:
            render_batch_result(st.session_state["batch_result"])

    with batch_right_column:
        with st.container(border=True):
            st.markdown("#### Как это работает")
            st.write(
                "Каждая непустая строка уходит в `/batch-analyze`, проходит одну "
                "batch-инференс-операцию и сохраняется как отдельная запись истории."
            )

        with st.container(border=True):
            st.markdown("#### Что вы увидите")
            st.markdown("- таблицу по всем строкам с классом, уверенностью и `Prediction ID`")
            st.markdown("- общее время batch-запроса и среднее на запись")
            st.markdown("- распределение классов по всей пачке")

        with st.container(border=True):
            st.markdown("#### Подсказка по вводу")
            st.code(
                "Спасибо за быстрый ответ\n"
                "Когда будет доставка?\n"
                "После обновления приложение работает хуже",
                language="text",
            )

with reviewer_tab:
    render_section_heading(
        "Reviewer view",
        "История, детали и агрегаты",
    )

    with st.container(border=True):
        dashboard_controls_left, dashboard_controls_right = st.columns([1.25, 0.75])
        history_label_options = {
            "Все классы": None,
            "Жалоба": "complaint",
            "Вопрос": "question",
            "Похвала": "praise",
            "Прочее": "other",
        }

        selected_history_label_title = dashboard_controls_left.selectbox(
            "Фильтр истории по конкретному классу",
            options=list(history_label_options),
            key="history_label_filter",
        )
        refresh_requested = dashboard_controls_right.button(
            "Обновить данные",
            use_container_width=True,
        )

    if refresh_requested:
        st.caption("История и статистика заново запрошены из backend REST API.")

    history_payload, history_error = request_prediction_history(
        label=history_label_options[selected_history_label_title],
    )
    stats_payload, stats_error = request_prediction_stats()

    review_left_column, review_right_column = st.columns([1.2, 0.95], gap="large")

    with review_left_column:
        with st.container(border=True):
            st.markdown("#### История предсказаний")
            if history_error:
                st.error(history_error)
            elif history_payload is None:
                st.info("История предсказаний пока недоступна.")
            else:
                history_items = list(history_payload["items"])
                st.caption(
                    f"Показано {len(history_items)} из {history_payload['total']} последних записей."
                )
                if history_items:
                    history_frame = make_history_frame(history_items)
                    st.dataframe(
                        history_frame.style.format({"Уверенность": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )

                    history_items_by_id = {
                        str(item["id"]): item for item in history_items
                    }
                    selected_prediction_id = st.selectbox(
                        "Выберите запись для подробностей",
                        options=list(history_items_by_id),
                        format_func=lambda prediction_id: (
                            f"{format_label_title(str(history_items_by_id[prediction_id]['label']))} · "
                            f"{format_timestamp(history_items_by_id[prediction_id]['created_at'])} · "
                            f"{shorten_text(str(history_items_by_id[prediction_id]['text']), limit=55)}"
                        ),
                        key="selected_prediction_id",
                    )

                    detail_payload, detail_error = request_prediction_detail(selected_prediction_id)
                    if detail_error:
                        st.error(detail_error)
                    elif detail_payload is not None:
                        render_prediction_detail(detail_payload)
                else:
                    if selected_history_label_title == "Все классы":
                        st.info(
                            "История пока пуста. Выполните single- или batch-анализ, чтобы появились записи."
                        )
                    else:
                        st.info(
                            "Для текущего фильтра истории записей пока нет. "
                            "Смените фильтр или выполните новый анализ."
                        )

    with review_right_column:
        with st.container(border=True):
            st.markdown("#### Агрегированная статистика")
            if stats_error:
                st.error(stats_error)
            elif stats_payload is None:
                st.info("Статистика пока недоступна.")
            else:
                total_predictions = int(stats_payload["total_predictions"])
                average_confidence = stats_payload.get("average_confidence")
                average_processing_time_ms = stats_payload.get("average_processing_time_ms")
                last_prediction_at = stats_payload.get("last_prediction_at")

                render_stat_cards(
                    [
                        {"label": "Всего", "value": str(total_predictions)},
                        {
                            "label": "Средняя уверенность",
                            "value": (
                                "n/a"
                                if average_confidence is None
                                else f"{float(average_confidence) * 100:.1f}%"
                            ),
                        },
                        {
                            "label": "Среднее, ms",
                            "value": (
                                "n/a"
                                if average_processing_time_ms is None
                                else f"{float(average_processing_time_ms):.1f}"
                            ),
                        },
                    ]
                )

                st.caption(
                    f"Последнее сохранённое предсказание: {format_timestamp(last_prediction_at)}"
                )

                if total_predictions == 0:
                    st.info(
                        "Пока нет сохранённых предсказаний, поэтому распределение классов ещё пустое."
                    )
                else:
                    st.markdown("**Распределение классов**")
                    render_stats_distribution_chart(dict(stats_payload["count_by_label"]))
