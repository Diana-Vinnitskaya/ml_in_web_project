from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
PROMETHEUS_PATH = REPO_ROOT / "monitoring" / "prometheus.yml"
GRAFANA_DATASOURCE_PATH = (
    REPO_ROOT
    / "monitoring"
    / "grafana"
    / "provisioning"
    / "datasources"
    / "prometheus.yml"
)
GRAFANA_DASHBOARD_PROVIDER_PATH = (
    REPO_ROOT
    / "monitoring"
    / "grafana"
    / "provisioning"
    / "dashboards"
    / "dashboard.yml"
)
GRAFANA_DASHBOARD_PATH = (
    REPO_ROOT
    / "monitoring"
    / "grafana"
    / "provisioning"
    / "dashboards"
    / "ru-feedback-dashboard.json"
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_only_nginx_publishes_host_ports() -> None:
    compose = load_yaml(COMPOSE_PATH)
    services = compose["services"]

    for service_name, service_config in services.items():
        published_ports = service_config.get("ports") or []
        if service_name == "nginx":
            assert published_ports, "nginx must publish the public host port."
            assert len(published_ports) == 1
            assert "80" in str(published_ports[0])
        else:
            assert not published_ports, (
                f"{service_name} must stay internal and should not publish host ports."
            )


def test_compose_monitoring_wiring_matches_operational_plan() -> None:
    compose = load_yaml(COMPOSE_PATH)
    services = compose["services"]

    backend_networks = set(services["backend"]["networks"])
    assert {"frontend_net", "backend_net", "monitoring_net"} <= backend_networks

    prometheus_networks = set(services["prometheus"]["networks"])
    grafana_networks = set(services["grafana"]["networks"])
    nginx_networks = set(services["nginx"]["networks"])

    assert prometheus_networks == {"monitoring_net"}
    assert grafana_networks == {"monitoring_net"}
    assert {"frontend_net", "monitoring_net"} <= nginx_networks


def test_prometheus_scrapes_backend_metrics_endpoint() -> None:
    config = load_yaml(PROMETHEUS_PATH)

    scrape_jobs = config["scrape_configs"]
    backend_job = next(
        (
            job
            for job in scrape_jobs
            if job.get("job_name") == "rufeedback-backend"
        ),
        None,
    )

    assert backend_job is not None, "Prometheus must define the backend scrape job."
    assert backend_job["metrics_path"] == "/metrics"
    assert backend_job["static_configs"][0]["targets"] == ["backend:8000"]


def test_grafana_provisioning_uses_prometheus_datasource() -> None:
    datasource_config = load_yaml(GRAFANA_DATASOURCE_PATH)
    providers_config = load_yaml(GRAFANA_DASHBOARD_PROVIDER_PATH)

    datasource = datasource_config["datasources"][0]
    provider = providers_config["providers"][0]

    assert datasource["name"] == "Prometheus"
    assert datasource["uid"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert provider["options"]["path"] == "/etc/grafana/provisioning/dashboards"
    assert GRAFANA_DASHBOARD_PATH.exists(), "The provisioned Grafana dashboard must exist."


def test_grafana_dashboard_contains_required_operational_panels() -> None:
    dashboard = json.loads(GRAFANA_DASHBOARD_PATH.read_text(encoding="utf-8"))
    panel_titles = {panel.get("title") for panel in dashboard.get("panels", [])}

    assert dashboard["uid"] == "ru-feedback-overview"
    assert dashboard["title"] == "RuFeedback Overview"
    assert {
        "HTTP Request Rate",
        "HTTP Status Codes",
        "HTTP Request Latency P95",
        "Prediction Throughput",
        "Prediction Duration P95",
    } <= panel_titles
