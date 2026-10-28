"""Avaliação final do modelo, feita uma única vez no conjunto de teste."""

from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: ClassifierMixin, features: pd.DataFrame, target: pd.Series
) -> dict[str, Any]:
    """Calcula as métricas de classificação e a matriz de confusão.

    As métricas de rótulo usam o limiar já embutido no modelo; a ROC-AUC usa
    as probabilidades e, por isso, não depende do limiar.

    Returns:
        Dicionário com as métricas e a matriz de confusão (tn, fp, fn, tp).
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
