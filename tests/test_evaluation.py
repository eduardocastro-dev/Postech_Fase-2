import numpy as np
import pandas as pd
import pytest

from purchase_intention.modeling.evaluation import evaluate_model


class _FixedModel:
    # Modelo falso com previsões conhecidas, para conferir as contas à mão.
    def __init__(self, probabilities: list[float], threshold: float) -> None:
        self._probabilities = np.array(probabilities)
        self._threshold = threshold

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self._probabilities >= self._threshold).astype(int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack([1 - self._probabilities, self._probabilities])


def test_metrics_match_hand_calculation() -> None:
    target = pd.Series([0, 0, 0, 1, 1])
    # Previsões com limiar 0.5: [0, 1, 0, 1, 0] -> tn=2, fp=1, fn=1, tp=1.
    model = _FixedModel([0.1, 0.6, 0.2, 0.9, 0.4], threshold=0.5)

    metrics = evaluate_model(model, pd.DataFrame(index=target.index), target)

    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.6)
    # Pares (positivo, negativo) ordenados corretamente: 5 de 6.
    assert metrics["roc_auc"] == pytest.approx(5 / 6)
