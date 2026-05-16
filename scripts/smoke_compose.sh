#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log() {
  printf '[smoke] %s\n' "$*"
}

fail() {
  printf '[smoke] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command is missing: $1"
  fi
}

load_env_file() {
  local env_file="$1"
  local line
  local key
  local value

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"
    export "$key=$value"
  done < "$env_file"
}

if [[ ! -f .env && -f .env.example ]]; then
  log "Creating .env from .env.example for local smoke validation"
  cp .env.example .env
fi

if [[ -f .env ]]; then
  load_env_file .env
fi

BASE_URL="${BASE_URL:-http://localhost:${NGINX_PORT:-80}}"
BASE_URL="${BASE_URL%/}"
API_BASE_URL="${API_BASE_URL:-${BASE_URL}/api/v1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-180}"
PROMETHEUS_SCRAPE_TIMEOUT_SECONDS="${PROMETHEUS_SCRAPE_TIMEOUT_SECONDS:-60}"

wait_for_status() {
  local url="$1"
  local expected_status="$2"
  local timeout_seconds="$3"
  local started_at
  local now
  local code

  started_at="$(date +%s)"
  while true; do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$REQUEST_TIMEOUT" "$url" || true)"
    if [[ "$code" == "$expected_status" ]]; then
      return 0
    fi

    now="$(date +%s)"
    if (( now - started_at >= timeout_seconds )); then
      fail "Timed out waiting for ${url} to return HTTP ${expected_status}. Last status: ${code:-unavailable}"
    fi
    sleep 3
  done
}

published_port() {
  local service="$1"
  local port="$2"
  local container_id

  container_id="$(docker compose ps -q "$service" 2>/dev/null || true)"
  if [[ -z "$container_id" ]]; then
    return 0
  fi

  docker inspect \
    "$container_id" \
    --format "{{with index .NetworkSettings.Ports \"${port}/tcp\"}}{{(index . 0).HostIp}}:{{(index . 0).HostPort}}{{end}}" \
    2>/dev/null || true
}

assert_only_nginx_publishes_host_ports() {
  local nginx_port_output
  nginx_port_output="$(published_port nginx 80)"
  if [[ -z "$nginx_port_output" ]]; then
    fail "Nginx must publish a host port for port 80"
  fi

  while read -r service port; do
    [[ -n "$service" ]] || continue
    if [[ -n "$(published_port "$service" "$port")" ]]; then
      fail "Service ${service} unexpectedly publishes host port ${port}"
    fi
  done <<'EOF'
backend 8000
ui 8501
postgres 5432
prometheus 9090
grafana 3000
EOF
}

assert_prediction_payload() {
  local payload="$1"

  PREDICTION_ID="$(
    PAYLOAD="$payload" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["PAYLOAD"])
required = {
    "id",
    "text",
    "label",
    "confidence",
    "probabilities",
    "processing_time_ms",
    "created_at",
}
missing = sorted(required - set(payload))
if missing:
    raise SystemExit(f"missing fields: {missing}")
print(payload["id"])
PY
  )"
}

assert_prediction_detail() {
  local payload="$1"
  local expected_id="$2"

  PAYLOAD="$payload" EXPECTED_ID="$expected_id" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["PAYLOAD"])
expected_id = os.environ["EXPECTED_ID"]
if payload.get("id") != expected_id:
    raise SystemExit(f"unexpected prediction id: {payload.get('id')} != {expected_id}")
if "label" not in payload or "probabilities" not in payload:
    raise SystemExit("prediction detail response is incomplete")
PY
}

require_command curl
require_command docker
require_command python3

log "Starting Docker Compose stack"
docker compose up --build -d
docker compose ps

log "Verifying only Nginx publishes a host port"
assert_only_nginx_publishes_host_ports

log "Waiting for Nginx, liveness, readiness, and aggregate health endpoints"
wait_for_status "${BASE_URL}/" "200" "$STARTUP_TIMEOUT_SECONDS"
wait_for_status "${API_BASE_URL}/health/live" "200" "$STARTUP_TIMEOUT_SECONDS"
wait_for_status "${API_BASE_URL}/health/ready" "200" "$STARTUP_TIMEOUT_SECONDS"
wait_for_status "${API_BASE_URL}/health" "200" "$STARTUP_TIMEOUT_SECONDS"

log "Validating single prediction flow through Nginx"
prediction_payload="$(
  curl -sS \
    --max-time "$REQUEST_TIMEOUT" \
    -X POST "${API_BASE_URL}/analyze" \
    -H 'Content-Type: application/json' \
    -d '{"text":"Доставка задержалась, но спасибо за честный ответ поддержки"}'
)"
assert_prediction_payload "$prediction_payload"

log "Checking rate limiting on /api/"
tmp_dir="$(mktemp -d)"
codes_file="${tmp_dir}/status_codes.txt"
seq 1 30 | xargs -I{} -P 30 sh -c \
  'curl -sS -o /dev/null -w "%{http_code}\n" --max-time "$1" "$2"' _ \
  "$REQUEST_TIMEOUT" "${API_BASE_URL}/health/live" > "$codes_file"

if ! grep -q '^429$' "$codes_file"; then
  rm -rf "$tmp_dir"
  fail "Expected at least one HTTP 429 response from Nginx rate limiting"
fi
rm -rf "$tmp_dir"
sleep 2

log "Checking direct backend /metrics exposition"
docker compose exec -T backend python - <<'PY'
import urllib.request

payload = urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode("utf-8")
required = [
    "rufeedback_http_requests_total",
    "rufeedback_http_request_duration_seconds",
    "rufeedback_predictions_total",
    "rufeedback_prediction_duration_seconds",
]
missing = [name for name in required if name not in payload]
if missing:
    raise SystemExit(f"Metrics endpoint is missing expected series: {missing}")
PY

log "Checking Prometheus target health and scraped prediction metrics"
docker compose exec -T backend python - <<'PY'
import json
import os
import time
import urllib.parse
import urllib.request

base_url = "http://prometheus:9090"
targets = json.load(urllib.request.urlopen(f"{base_url}/api/v1/targets"))
active_targets = targets["data"]["activeTargets"]
if not any(
    target.get("labels", {}).get("job") == "rufeedback-backend"
    and target.get("health") == "up"
    for target in active_targets
):
    raise SystemExit(f"Prometheus target is not healthy: {active_targets}")

query = urllib.parse.quote("sum(rufeedback_predictions_total)")
deadline = time.time() + int(os.environ.get("PROMETHEUS_SCRAPE_TIMEOUT_SECONDS", "60"))
last_result = None
while time.time() < deadline:
    result = json.load(urllib.request.urlopen(f"{base_url}/api/v1/query?query={query}"))
    last_result = result
    series = result["data"]["result"]
    if series:
        break
    time.sleep(3)
else:
    raise SystemExit(f"No scraped prediction metrics found before timeout: {last_result}")
PY

log "Checking Grafana dashboard provisioning and panel availability"
docker compose exec \
  -T \
  -e GRAFANA_USER="${GRAFANA_ADMIN_USER:-admin}" \
  -e GRAFANA_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}" \
  backend python - <<'PY'
import base64
import json
import os
import urllib.request

credentials = f"{os.environ['GRAFANA_USER']}:{os.environ['GRAFANA_PASSWORD']}"
token = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
request = urllib.request.Request("http://grafana:3000/api/dashboards/uid/ru-feedback-overview")
request.add_header("Authorization", f"Basic {token}")
payload = json.load(urllib.request.urlopen(request))
dashboard = payload["dashboard"]
panel_titles = {panel.get("title") for panel in dashboard.get("panels", [])}
required_titles = {
    "HTTP Request Rate",
    "HTTP Status Codes",
    "HTTP Request Latency P95",
    "Prediction Throughput",
    "Prediction Duration P95",
}
missing = sorted(required_titles - panel_titles)
if missing:
    raise SystemExit(f"Grafana dashboard is missing expected panels: {missing}")
PY

log "Restarting backend to verify PostgreSQL-backed persistence"
docker compose restart backend
wait_for_status "${API_BASE_URL}/health/ready" "200" "$STARTUP_TIMEOUT_SECONDS"

prediction_detail="$(
  curl -sS \
    --max-time "$REQUEST_TIMEOUT" \
    "${API_BASE_URL}/predictions/${PREDICTION_ID}"
)"
assert_prediction_detail "$prediction_detail" "$PREDICTION_ID"

log "Smoke checks passed"
