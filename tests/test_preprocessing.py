import numpy as np
import pandas as pd
import pytest

from purchase_intention.layers.schema import FEATURE_COLUMNS, SESSION_ID, TARGET
from purchase_intention.modeling.preprocessing import (
    SKEWED_COLUMNS,
    build_preprocessor,
    split_features_target,
)


def test_split_features_target_drops_id_and_target(
    gold_sessions: pd.DataFrame,
) -> None:
    features, target = split_features_target(gold_sessions)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert SESSION_ID not in features.columns
    assert target.name == TARGET


def test_skewed_columns_exclude_rates() -> None:
    assert "PageValues" in SKEWED_COLUMNS
    assert "BounceRates" not in SKEWED_COLUMNS


def test_output_is_numeric_and_without_missing_values(
    gold_sessions: pd.DataFrame,
) -> None:
    features, _ = split_features_target(gold_sessions)

    transformed = build_preprocessor().fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert np.issubdtype(transformed.dtype, np.number)
    assert not np.isnan(transformed).any()


def test_unknown_category_does_not_break_prediction(
    gold_sessions: pd.DataFrame,
) -> None:
    features, _ = split_features_target(gold_sessions)
    preprocessor = build_preprocessor().fit(features)
    unseen = features.head(1).assign(Browser="999")

    transformed = preprocessor.transform(unseen)

    # A categoria nova vira um vetor só de zeros nas colunas de Browser.
    names = preprocessor.get_feature_names_out()
    browser_columns = [i for i, name in enumerate(names) if "Browser_" in name]
    assert not transformed[0, browser_columns].any()


def test_fit_learns_only_from_given_data(gold_sessions: pd.DataFrame) -> None:
    features, _ = split_features_target(gold_sessions)
    half = len(features) // 2

    preprocessor = build_preprocessor().fit(features.iloc[:half])

    # A média do scaler vem só da primeira metade (sem vazamento do restante).
    scaler = preprocessor.named_transformers_["rates"]
    expected = features.iloc[:half]["BounceRates"].mean()
    assert scaler.mean_[0] == pytest.approx(expected)
