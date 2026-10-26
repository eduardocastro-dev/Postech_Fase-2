from pathlib import Path

import pytest
from pydantic import ValidationError

from purchase_intention.config import Settings

SETTINGS_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove variáveis que possam existir na máquina, para que cada teste
    # parta de um ambiente conhecido.
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_used_without_env_file() -> None:
    # _env_file=None ignora um eventual .env local do desenvolvedor.
    settings = Settings(_env_file=None)

    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_experiment_name == "purchase-intention"
    assert settings.mlflow_registered_model_name == "purchase-intention-classifier"
    assert settings.log_level == "INFO"


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.mlflow_tracking_uri == "http://mlflow-server:5000"
    assert settings.log_level == "DEBUG"


def test_env_file_values_are_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_EXPERIMENT_NAME=from-env-file\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.mlflow_experiment_name == "from-env-file"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
