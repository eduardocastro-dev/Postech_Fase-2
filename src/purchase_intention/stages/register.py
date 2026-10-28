"""Estágio 6: publicação do modelo no MLflow (Tracking e Model Registry).

Uso: ``python -m purchase_intention.stages.register``

Roda em todo ``dvc repro`` (``always_changed``), mas só grava algo quando o
modelo ainda não foi publicado no MLflow configurado.
"""

import json

import mlflow
import pandas as pd

from purchase_intention.config import get_settings
from purchase_intention.modeling.preprocessing import split_features_target
from purchase_intention.params import PipelineParams
from purchase_intention.stages.common import bootstrap
from purchase_intention.tracking.publisher import ModelPublisher, TrainingOutputs

# Quantidade de sessões guardadas como exemplo de entrada do modelo.
INPUT_EXAMPLE_ROWS = 5


def build_outputs(params: PipelineParams) -> TrainingOutputs:
    """Reúne os arquivos do treino e os parâmetros que serão publicados."""
    metadata = json.loads(params.data.bronze_metadata.read_text(encoding="utf-8"))
    test = pd.read_parquet(params.data.gold_test).head(INPUT_EXAMPLE_ROWS)
    return TrainingOutputs(
        model_path=params.artifacts.model,
        cv_results_path=params.artifacts.cv_results,
        metrics_path=params.artifacts.metrics,
        input_example=split_features_target(test)[0],
        run_params={
            "seed": params.seed,
            "cv_folds": params.train.cv_folds,
            "test_size": params.split.test_size,
            "data_sha256": metadata["source_sha256"],
        },
        hyperparameters=params.train.models,
    )


def main() -> None:
    """Publica o modelo treinado no MLflow configurado em ``.env``."""
    params = bootstrap()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    ModelPublisher(
        experiment_name=settings.mlflow_experiment_name,
        registered_model_name=settings.mlflow_registered_model_name,
    ).publish(build_outputs(params))


if __name__ == "__main__":
    main()
