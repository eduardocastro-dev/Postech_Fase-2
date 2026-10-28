import numpy as np
import pandas as pd
import pytest

from purchase_intention.modeling.candidates import build_candidate
from purchase_intention.modeling.preprocessing import split_features_target
from purchase_intention.modeling.selection import (
    SCORING,
    CandidateScore,
    make_cv,
    out_of_fold_probabilities,
    score_candidate,
    select_best,
    tune_threshold,
)


def _score(name: str, f1: float, roc_auc: float) -> CandidateScore:
    return CandidateScore(name=name, mean={"f1": f1, "roc_auc": roc_auc}, std={})


def test_select_best_prefers_highest_f1() -> None:
    scores = [_score("a", 0.60, 0.95), _score("b", 0.65, 0.80)]

    assert select_best(scores).name == "b"


def test_select_best_uses_roc_auc_to_break_ties() -> None:
    scores = [_score("a", 0.65, 0.85), _score("b", 0.65, 0.90)]

    assert select_best(scores).name == "b"


def test_tune_threshold_maximizes_f1() -> None:
    target = pd.Series([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.35, 0.9])

    # Com limiar 0.35 as duas compras são capturadas sem falsos positivos.
    assert tune_threshold(target, probabilities) == pytest.approx(0.35)


def test_score_candidate_reports_every_metric(gold_sessions: pd.DataFrame) -> None:
    features, target = split_features_target(gold_sessions)
    pipeline = build_candidate("logistic_regression", {"max_iter": 500}, seed=0)

    score = score_candidate("lr", pipeline, features, target, make_cv(3, seed=0))

    assert set(score.mean) == set(SCORING)
    assert all(0.0 <= value <= 1.0 for value in score.mean.values())


def test_out_of_fold_gives_one_probability_per_session(
    gold_sessions: pd.DataFrame,
) -> None:
    features, target = split_features_target(gold_sessions)
    pipeline = build_candidate("logistic_regression", {"max_iter": 500}, seed=0)

    probabilities = out_of_fold_probabilities(
        pipeline, features, target, make_cv(3, seed=0)
    )

    assert probabilities.shape == (len(features),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
