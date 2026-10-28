"""Gravação das execuções no MLflow Tracking.

Cada modelo publicado gera uma execução principal (a seleção, com o vencedor e
as métricas de teste) e uma execução filha por candidato da validação cruzada.
"""

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
from sklearn.base import ClassifierMixin

MODEL_ARTIFACT_NAME = "model"
# Pacotes necessários para carregar o modelo (serializado em skops, o padrão
# do MLflow 3). Declarados explicitamente: a inferência automática é lenta e
# incluía pacotes do ambiente de desenvolvimento, como o pytest.
MODEL_PACKAGES = ("scikit-learn", "numpy", "scipy", "pandas", "skops")


def model_requirements() -> list[str]:
    """Fixa os pacotes do modelo nas versões instaladas no ambiente atual."""
    return [f"{package}=={version(package)}" for package in MODEL_PACKAGES]


def flatten_cv_metrics(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Converte médias e desvios da validação cruzada em métricas do MLflow."""
    metrics = {f"cv_{name}": value for name, value in scores["mean"].items()}
    metrics |= {f"cv_{name}_std": value for name, value in scores["std"].items()}
    return metrics


def flatten_test_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Converte o ``metrics.json`` em métricas do MLflow, com prefixo ``test_``."""
    flat = {
        f"test_{name}": value
        for name, value in metrics.items()
        if name != "confusion_matrix"
    }
    flat |= {
        f"test_{cell}": count for cell, count in metrics["confusion_matrix"].items()
    }
    return flat


def log_selection(
    run_params: Mapping[str, Any],
    cv_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    report_paths: list[Path],
) -> None:
    """Registra na execução ativa os parâmetros, as métricas e os relatórios."""
    selected = cv_report["selected_model"]
    mlflow.log_params(
        {**run_params, "selected_model": selected, "threshold": cv_report["threshold"]}
    )
    mlflow.log_metrics(flatten_cv_metrics(cv_report["candidates"][selected]))
    mlflow.log_metrics(flatten_test_metrics(metrics))
    for path in report_paths:
        mlflow.log_artifact(str(path), artifact_path="reports")


def log_candidates(
    cv_report: Mapping[str, Any],
    hyperparameters: Mapping[str, Mapping[str, Any]],
) -> None:
    """Cria uma execução filha por candidato, com hiperparâmetros e métricas."""
    for name, scores in cv_report["candidates"].items():
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params({"model": name, **hyperparameters.get(name, {})})
            mlflow.log_metrics(flatten_cv_metrics(scores))


def trusted_types(model: ClassifierMixin) -> list[str]:
    """Tipos internos do modelo que o skops não carrega sem autorização explícita.

    Por segurança, o skops recusa tipos fora da sua lista padrão (por exemplo,
    as árvores do Random Forest). Autorizamos apenas os tipos deste modelo,
    que é gerado pelo próprio pipeline, nunca recebido de terceiros.
    """
    return skops.io.get_untrusted_types(data=skops.io.dumps(model))


def log_model(model: ClassifierMixin, input_example: pd.DataFrame) -> ModelInfo:
    """Grava o modelo na execução ativa, com assinatura e exemplo de entrada."""
    signature = infer_signature(input_example, model.predict(input_example))
    return mlflow.sklearn.log_model(
        model,
        name=MODEL_ARTIFACT_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=model_requirements(),
        skops_trusted_types=trusted_types(model),
    )
