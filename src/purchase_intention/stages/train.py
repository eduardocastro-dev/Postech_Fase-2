"""Estágio 4: seleção e treino do modelo com o conjunto de treino da Gold.

Uso: ``python -m purchase_intention.stages.train``
"""

import logging

import pandas as pd

from purchase_intention.layers.storage import write_json
from purchase_intention.modeling.artifacts import save_model
from purchase_intention.modeling.training import build_cv_report, train_best_model
from purchase_intention.stages.common import bootstrap

logger = logging.getLogger(__name__)


def main() -> None:
    """Executa a seleção e grava o modelo final e o relatório da validação."""
    params = bootstrap()
    train = pd.read_parquet(params.data.gold_train)
    result = train_best_model(train, params.train, params.seed)
    save_model(result.model, params.artifacts.model)
    write_json(
        build_cv_report(result, params.train.cv_folds), params.artifacts.cv_results
    )
    logger.info(
        "[train] Modelo escolhido: %s (F1 médio %.4f, limiar %.4f)",
        result.selected.name,
        result.selected.mean["f1"],
        result.threshold,
    )


if __name__ == "__main__":
    main()
