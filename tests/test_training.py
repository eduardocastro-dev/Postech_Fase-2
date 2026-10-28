from pathlib import Path

import numpy as np
import pandas as pd

from purchase_intention.modeling.artifacts import load_model, save_model
from purchase_intention.modeling.preprocessing import split_features_target
from purchase_intention.modeling.training import build_cv_report, train_best_model
from purchase_intention.params import TrainParams


def test_selected_model_is_the_best_scored(
    gold_sessions: pd.DataFrame, small_train_params: TrainParams
) -> None:
    result = train_best_model(gold_sessions, small_train_params, seed=0)

    best_f1 = max(score.mean["f1"] for score in result.scores)
    assert result.selected.mean["f1"] == best_f1
    assert len(result.scores) == len(small_train_params.models)


def test_final_model_applies_tuned_threshold(
    gold_sessions: pd.DataFrame, small_train_params: TrainParams
) -> None:
    result = train_best_model(gold_sessions, small_train_params, seed=0)
    features, _ = split_features_target(gold_sessions)

    probabilities = result.model.predict_proba(features)[:, 1]
    expected = (probabilities >= result.threshold).astype(int)
    np.testing.assert_array_equal(result.model.predict(features), expected)


def test_same_seed_reproduces_training(
    gold_sessions: pd.DataFrame, small_train_params: TrainParams
) -> None:
    first = train_best_model(gold_sessions, small_train_params, seed=0)
    repeated = train_best_model(gold_sessions, small_train_params, seed=0)

    assert first.selected == repeated.selected
    assert first.threshold == repeated.threshold


def test_cv_report_lists_every_candidate(
    gold_sessions: pd.DataFrame, small_train_params: TrainParams
) -> None:
    result = train_best_model(gold_sessions, small_train_params, seed=0)

    report = build_cv_report(result, cv_folds=small_train_params.cv_folds)

    assert report["selected_model"] == result.selected.name
    assert set(report["candidates"]) == set(small_train_params.models)
    assert set(report["candidates"]["random_forest"]) == {"mean", "std"}


def test_saved_model_predicts_the_same(
    gold_sessions: pd.DataFrame, small_train_params: TrainParams, tmp_path: Path
) -> None:
    result = train_best_model(gold_sessions, small_train_params, seed=0)
    features, _ = split_features_target(gold_sessions)
    path = tmp_path / "models" / "model.joblib"

    save_model(result.model, path)

    loaded = load_model(path)
    np.testing.assert_array_equal(
        loaded.predict(features), result.model.predict(features)
    )
