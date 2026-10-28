"""Publicação idempotente do modelo treinado no MLflow.

O treino acontece no DVC; aqui apenas publicamos o que ele produziu. O modelo
é identificado pelo SHA-256 do arquivo: se esse hash já estiver no MLflow,
nada é gravado de novo. Assim, um clone recém-feito preenche o MLflow local
sem retreinar, e execuções repetidas não criam duplicatas.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from purchase_intention.layers.storage import sha256_of_file
from purchase_intention.modeling.artifacts import load_model
from purchase_intention.tracking.registry import promote_if_better, register_version
from purchase_intention.tracking.runs import log_candidates, log_model, log_selection

MODEL_SHA256_TAG = "model_sha256"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingOutputs:
    """Tudo o que o treino produziu e que será publicado."""

    model_path: Path
    cv_results_path: Path
    metrics_path: Path
    input_example: pd.DataFrame
    run_params: Mapping[str, Any]
    hyperparameters: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Publication:
    """Resultado da publicação de uma nova versão."""

    run_id: str
    version: str
    is_champion: bool


class ModelPublisher:
    """Publica execuções, registra a versão do modelo e promove o campeão."""

    def __init__(
        self,
        experiment_name: str,
        registered_model_name: str,
        artifact_location: str | None = None,
    ) -> None:
        """Inicializa o publicador.

        Args:
            experiment_name: Experimento do MLflow (criado se não existir).
            registered_model_name: Nome do modelo no Model Registry.
            artifact_location: Onde guardar os artefatos de um experimento
                novo. ``None`` usa o padrão do servidor ou do MLflow local.
        """
        self._experiment_name = experiment_name
        self._registered_model_name = registered_model_name
        self._artifact_location = artifact_location

    def publish(self, outputs: TrainingOutputs) -> Publication | None:
        """Publica o modelo, a menos que ele já tenha sido publicado.

        Returns:
            Dados da nova versão, ou ``None`` se o modelo já estava no MLflow.
        """
        experiment_id = self._ensure_experiment()
        model_sha256 = sha256_of_file(outputs.model_path)
        if self._is_published(experiment_id, model_sha256):
            logger.info("[register] Modelo %s já publicado", model_sha256[:12])
            return None
        return self._publish_new(experiment_id, model_sha256, outputs)

    def _ensure_experiment(self) -> str:
        """Ativa o experimento, criando-o na primeira execução, e devolve o id.

        Ativar é necessário para que as execuções filhas caiam no mesmo
        experimento da principal.
        """
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self._experiment_name, artifact_location=self._artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)
        return experiment_id

    def _is_published(self, experiment_id: str, model_sha256: str) -> bool:
        """Procura uma publicação completa deste modelo.

        Só conta execução ``FINISHED``: se algo falhou no meio, a execução
        termina como ``FAILED`` e a próxima rodada publica de novo.
        """
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.{MODEL_SHA256_TAG} = '{model_sha256}'"
                " AND attributes.status = 'FINISHED'"
            ),
            max_results=1,
            output_format="list",
        )
        return bool(runs)

    def _publish_new(
        self, experiment_id: str, model_sha256: str, outputs: TrainingOutputs
    ) -> Publication:
        """Grava as execuções e registra a versão, tudo dentro da mesma execução."""
        cv_report = _read_json(outputs.cv_results_path)
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"selection-{model_sha256[:8]}",
            tags={MODEL_SHA256_TAG: model_sha256},
        ) as run:
            log_selection(
                outputs.run_params,
                cv_report,
                _read_json(outputs.metrics_path),
                [outputs.cv_results_path, outputs.metrics_path],
            )
            log_candidates(cv_report, outputs.hyperparameters)
            model_info = log_model(
                load_model(outputs.model_path), outputs.input_example
            )
            # O registro fica dentro da execução: se falhar, ela termina FAILED.
            return self._register(run.info.run_id, model_info.model_uri, cv_report)

    def _register(
        self, run_id: str, model_uri: str, cv_report: Mapping[str, Any]
    ) -> Publication:
        """Registra a nova versão e decide se ela vira a campeã."""
        selected = cv_report["selected_model"]
        cv_f1 = cv_report["candidates"][selected]["mean"]["f1"]
        name = self._registered_model_name
        version = register_version(model_uri, name, cv_f1)
        is_champion = promote_if_better(MlflowClient(), name, version, cv_f1)
        logger.info(
            "[register] %s versão %s publicada (campeã: %s)", name, version, is_champion
        )
        return Publication(run_id=run_id, version=version, is_champion=is_champion)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
