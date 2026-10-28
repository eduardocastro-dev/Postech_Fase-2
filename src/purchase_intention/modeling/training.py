"""Treino completo: seleção do candidato, ajuste do limiar e ajuste final."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import FixedThresholdClassifier

from purchase_intention.modeling.candidates import build_candidates
from purchase_intention.modeling.preprocessing import split_features_target
from purchase_intention.modeling.selection import (
    CandidateScore,
    make_cv,
    out_of_fold_probabilities,
    score_candidates,
    select_best,
    tune_threshold,
)
from purchase_intention.params import TrainParams

SELECTION_CRITERION = "f1 médio na validação cruzada (desempate: roc_auc)"


@dataclass(frozen=True)
class TrainingResult:
    """Modelo final e as informações que justificam a sua escolha."""

    model: FixedThresholdClassifier
    selected: CandidateScore
    scores: list[CandidateScore]
    threshold: float


def train_best_model(
    train: pd.DataFrame, params: TrainParams, seed: int
) -> TrainingResult:
    """Seleciona o melhor candidato e o retreina no conjunto de treino inteiro.

    O limiar vem das previsões *out-of-fold* do vencedor; o modelo final já
    aplica esse limiar em ``predict``.
    """
    features, target = split_features_target(train)
    cv = make_cv(params.cv_folds, seed)
    candidates = build_candidates(params.models, seed)
    scores = score_candidates(candidates, features, target, cv)
    selected = select_best(scores)
    winner = candidates[selected.name]
    probabilities = out_of_fold_probabilities(winner, features, target, cv)
    threshold = tune_threshold(target, probabilities)
    model = FixedThresholdClassifier(
        winner, threshold=threshold, response_method="predict_proba"
    ).fit(features, target)
    return TrainingResult(model, selected, scores, threshold)


def build_cv_report(result: TrainingResult, cv_folds: int) -> dict[str, Any]:
    """Monta o relatório da seleção, gravado em ``reports/cv_results.json``."""
    return {
        "criterion": SELECTION_CRITERION,
        "cv_folds": cv_folds,
        "selected_model": result.selected.name,
        "threshold": round(result.threshold, 6),
        "candidates": {
            score.name: {
                "mean": _rounded(score.mean),
                "std": _rounded(score.std),
            }
            for score in result.scores
        },
    }


def _rounded(values: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in values.items()}
