"""Seleção do modelo e do limiar de decisão, usando apenas o conjunto de treino.

Critério: maior F1 médio da classe positiva na validação cruzada; em caso de
empate, maior ROC-AUC. O conjunto de teste não participa de nenhuma escolha.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

SCORING = ("f1", "roc_auc", "accuracy", "precision", "recall")


@dataclass(frozen=True)
class CandidateScore:
    """Médias e desvios das métricas de um candidato na validação cruzada."""

    name: str
    mean: dict[str, float]
    std: dict[str, float]


def make_cv(folds: int, seed: int) -> StratifiedKFold:
    """Cria as dobras estratificadas, embaralhadas com semente fixa."""
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def score_candidate(
    name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
) -> CandidateScore:
    """Avalia um candidato com validação cruzada em todas as métricas."""
    results = cross_validate(
        pipeline, features, target, cv=cv, scoring=SCORING, error_score="raise"
    )
    return CandidateScore(
        name=name,
        mean={metric: float(np.mean(results[f"test_{metric}"])) for metric in SCORING},
        std={metric: float(np.std(results[f"test_{metric}"])) for metric in SCORING},
    )


def score_candidates(
    candidates: Mapping[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
) -> list[CandidateScore]:
    """Avalia todos os candidatos com as mesmas dobras."""
    return [
        score_candidate(name, pipeline, features, target, cv)
        for name, pipeline in candidates.items()
    ]


def select_best(scores: Iterable[CandidateScore]) -> CandidateScore:
    """Escolhe o maior F1 médio; a ROC-AUC média desempata."""
    return max(scores, key=lambda score: (score.mean["f1"], score.mean["roc_auc"]))


def out_of_fold_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
) -> np.ndarray:
    """Probabilidade da classe positiva para cada sessão, prevista sem tê-la visto.

    Cada sessão é prevista pelo modelo treinado nas outras dobras.
    """
    probabilities = cross_val_predict(
        pipeline, features, target, cv=cv, method="predict_proba"
    )
    return probabilities[:, 1]


def tune_threshold(target: pd.Series, probabilities: np.ndarray) -> float:
    """Encontra o limiar de probabilidade que maximiza o F1 da classe positiva."""
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    # O último ponto da curva (recall 0) não tem limiar associado.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[np.argmax(f1)])
