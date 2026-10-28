"""Valida a entrega em um clone temporário e independente do repositório.

Uso: python scripts/validate_clean_clone.py
Requer Git, Python 3.11, Poetry 2.x e acesso ao repositório informado.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path

DEFAULT_REPOSITORY = "https://github.com/eduardocastro-dev/Postech_Fase-2.git"
RAW_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
TEST_METRICS = {
    "f1": 0.6616541353383458,
    "roc_auc": 0.9235850810463165,
    "accuracy": 0.8905109489051095,
    "precision": 0.6346153846153846,
    "recall": 0.6910994764397905,
}


def run(command: list[str], cwd: Path) -> None:
    """Executa uma etapa de validação e interrompe em caso de falha."""
    print(f"\n[validacao] {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_REGISTERED_MODEL_NAME",
        "VIRTUAL_ENV",
        "POETRY_ACTIVE",
    ):
        environment.pop(name, None)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def require(condition: bool, message: str) -> None:
    """Falha explicitamente se uma condição do release não for atendida."""
    if not condition:
        raise RuntimeError(message)


def verify_outputs(project: Path) -> None:
    """Confere os artefatos, as métricas e a versão entregue do modelo."""
    raw_path = project / "data/raw/online_shoppers_intention.csv"
    require(
        hashlib.sha256(raw_path.read_bytes()).hexdigest() == RAW_SHA256,
        "SHA-256 do CSV divergente",
    )
    require((project / "models/model.joblib").is_file(), "Modelo ausente")
    report = json.loads((project / "reports/metrics.json").read_text(encoding="utf-8"))
    cv = json.loads((project / "reports/cv_results.json").read_text(encoding="utf-8"))
    for name, expected in TEST_METRICS.items():
        require(math.isclose(report[name], expected, abs_tol=1e-9), name)
    require(cv["selected_model"] == "random_forest", "Modelo selecionado divergente")
    require(math.isclose(cv["threshold"], 0.586776, abs_tol=1e-9), "Limiar divergente")
    require(sum(report["confusion_matrix"].values()) == 2466, "Total do teste")
    print("[validacao] Dados e métricas do release conferidos.", flush=True)


def verify_registry(project: Path) -> None:
    """Confirma a versão campeã, sem duplicação, e suas previsões."""
    snippet = textwrap.dedent(
        """
        import joblib
        import mlflow
        import mlflow.sklearn
        import numpy as np
        import pandas as pd
        from mlflow import MlflowClient
        from purchase_intention.config import get_settings
        from purchase_intention.modeling.preprocessing import split_features_target

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        champion = client.get_model_version_by_alias(
            settings.mlflow_registered_model_name, "champion"
        )
        versions = client.search_model_versions(
            f"name='{settings.mlflow_registered_model_name}'"
        )
        if str(champion.version) != "1" or len(versions) != 1:
            raise RuntimeError("Versão incorreta ou publicação duplicada no Registry")
        registered = mlflow.sklearn.load_model(
            f"models:/{settings.mlflow_registered_model_name}@champion"
        )
        original = joblib.load("models/model.joblib")
        sample = pd.read_parquet("data/gold/test.parquet").head(10)
        features, _ = split_features_target(sample)
        np.testing.assert_array_equal(
            registered.predict(features), original.predict(features)
        )
        print("[validacao] Champion v1 confere com o modelo versionado no DVC.")
        """
    )
    run(["poetry", "run", "python", "-c", snippet], project)


def validate(project: Path) -> None:
    """Executa a instalação, a qualidade e o pipeline em ambiente isolado."""
    run(["poetry", "install"], project)
    run(["poetry", "run", "ruff", "check", "."], project)
    run(["poetry", "run", "ruff", "format", "--check", "."], project)
    run(["poetry", "run", "pytest"], project)
    run(["poetry", "run", "dvc", "repro", "--pull"], project)
    verify_outputs(project)
    run(["poetry", "run", "dvc", "repro", "--pull"], project)
    verify_registry(project)
    run(["poetry", "run", "dvc", "metrics", "show"], project)


def _retry_readonly_removal(
    operation: Callable[[str], object], path: str, _error: object
) -> None:
    """Permite a remoção de arquivos do cache DVC protegidos no Windows."""
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    operation(path)


def cleanup_temporary_directory(directory: Path) -> None:
    """Remove o clone sem sobrescrever o resultado da validação."""
    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(directory, onexc=_retry_readonly_removal)
        else:
            shutil.rmtree(directory, onerror=_retry_readonly_removal)
    except OSError as error:
        print(
            f"[validacao] AVISO: limpeza incompleta de {directory}: {error}",
            file=sys.stderr,
            flush=True,
        )


def main() -> int:
    """Clona a referência indicada e verifica a entrega sem usar dados locais."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--keep", action="store_true", help="Mantém o clone criado")
    options = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="purchase-intention-release-"))
    project = directory / "Postech_Fase-2"
    try:
        command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            options.branch,
            options.repository,
            str(project),
        ]
        run(command, Path.cwd())
        validate(project)
        print("\n[validacao] Clone limpo aprovado.", flush=True)
        return 0
    except (subprocess.CalledProcessError, RuntimeError, OSError) as error:
        print(f"\n[validacao] FALHA: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if options.keep:
            print(f"[validacao] Clone preservado em: {project}", flush=True)
        else:
            cleanup_temporary_directory(directory)


if __name__ == "__main__":
    sys.exit(main())
