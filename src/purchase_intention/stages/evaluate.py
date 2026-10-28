"""Estágio 5: avaliação final do modelo no conjunto de teste da Gold.

Uso: ``python -m purchase_intention.stages.evaluate``
"""

import logging

import pandas as pd

from purchase_intention.layers.storage import write_json
from purchase_intention.modeling.artifacts import load_model
from purchase_intention.modeling.evaluation import evaluate_model
from purchase_intention.modeling.preprocessing import split_features_target
from purchase_intention.stages.common import bootstrap

logger = logging.getLogger(__name__)


def main() -> None:
    """Avalia o modelo treinado e grava as métricas de teste."""
    params = bootstrap()
    model = load_model(params.artifacts.model)
    features, target = split_features_target(pd.read_parquet(params.data.gold_test))
    metrics = evaluate_model(model, features, target)
    write_json(metrics, params.artifacts.metrics)
    logger.info(
        "[evaluate] Teste: F1 %.4f | ROC-AUC %.4f",
        metrics["f1"],
        metrics["roc_auc"],
    )


if __name__ == "__main__":
    main()
