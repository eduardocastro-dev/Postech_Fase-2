"""Testes estáticos da integração Docker Compose da Fase 8."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))
DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_has_multistage_build_and_non_root_runtime() -> None:
    assert "FROM python:3.11-slim AS builder" in DOCKERFILE
    assert "FROM python:3.11-slim AS runtime" in DOCKERFILE
    assert "poetry install --only main --no-root --sync" in DOCKERFILE
    assert "poetry install --only-root" in DOCKERFILE
    assert "USER app" in DOCKERFILE


def test_dockerfile_includes_local_dvc_storage_and_pipeline_files() -> None:
    for path in ("dvc-storage/", ".dvc/", "dvc.yaml", "dvc.lock", "data/", "src/"):
        assert path in DOCKERFILE


def test_compose_uses_two_services_without_external_credentials() -> None:
    services = COMPOSE["services"]
    assert set(services) == {"mlflow-server", "pipeline"}
    assert all("env_file" not in service for service in services.values())
    assert all("secrets" not in service for service in services.values())
    assert services["mlflow-server"]["image"] == services["pipeline"]["image"]


def test_mlflow_server_persists_database_and_artifacts() -> None:
    server = COMPOSE["services"]["mlflow-server"]
    assert "--backend-store-uri" in server["command"]
    assert "sqlite:////var/lib/mlflow/mlflow.db" in server["command"]
    assert "--artifacts-destination" in server["command"]
    assert "/var/lib/mlflow/artifacts" in server["command"]
    assert "mlflow-data:/var/lib/mlflow" in server["volumes"]
    assert "mlflow-data" in COMPOSE["volumes"]
    assert server["ports"] == ["127.0.0.1:5000:5000"]


def test_pipeline_waits_for_healthy_mlflow_and_pulls_dvc_data() -> None:
    pipeline = COMPOSE["services"]["pipeline"]
    assert COMPOSE["services"]["mlflow-server"]["healthcheck"]["test"]
    assert pipeline["depends_on"]["mlflow-server"]["condition"] == "service_healthy"
    assert pipeline["environment"]["MLFLOW_TRACKING_URI"] == (
        "http://mlflow-server:5000"
    )
    assert pipeline["command"] == ["dvc", "repro", "--pull"]
    assert pipeline["restart"] == "no"


def test_local_dvc_remote_contains_all_versioned_outputs() -> None:
    lock = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8"))
    raw = yaml.safe_load(
        (PROJECT_ROOT / "data/raw/online_shoppers_intention.csv.dvc").read_text(
            encoding="utf-8"
        )
    )
    stages = lock["stages"].values()
    md5_hashes = {output["md5"] for stage in stages for output in stage.get("outs", [])}
    md5_hashes.update(output["md5"] for output in raw["outs"])
    remote = PROJECT_ROOT / "dvc-storage" / "files" / "md5"

    assert md5_hashes
    for md5 in md5_hashes:
        assert (remote / md5[:2] / md5[2:]).is_file(), md5
