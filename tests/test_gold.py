from pathlib import Path

import pandas as pd
import pytest

from purchase_intention.layers.gold import GoldBuilder, GoldDatasets
from purchase_intention.layers.schema import SESSION_ID, TARGET, DataValidationError
from purchase_intention.layers.silver import SilverTransformer


@pytest.fixture
def silver_sessions(raw_sessions: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    # Usa a própria Silver para gerar a entrada, garantindo o mesmo formato
    # que a Gold recebe no pipeline real.
    transformer = SilverTransformer(tmp_path / "in", tmp_path / "out", tmp_path / "r")
    return transformer.transform(raw_sessions)


def _make_builder(tmp_path: Path, seed: int = 42) -> GoldBuilder:
    return GoldBuilder(
        input_path=tmp_path / "silver.parquet",
        train_path=tmp_path / "gold" / "train.parquet",
        test_path=tmp_path / "gold" / "test.parquet",
        test_size=0.2,
        seed=seed,
    )


def test_split_respects_test_size(
    silver_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    result = _make_builder(tmp_path).transform(silver_sessions)

    assert len(result.train) == 160
    assert len(result.test) == 40


def test_split_is_stratified_and_without_overlap(
    silver_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    result = _make_builder(tmp_path).transform(silver_sessions)

    assert result.train[TARGET].mean() == pytest.approx(0.2)
    assert result.test[TARGET].mean() == pytest.approx(0.2)
    assert not set(result.train[SESSION_ID]) & set(result.test[SESSION_ID])


def test_target_is_converted_to_zero_or_one(
    silver_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    result = _make_builder(tmp_path).transform(silver_sessions)

    assert pd.api.types.is_integer_dtype(result.train[TARGET])
    assert set(result.train[TARGET]) == {0, 1}


def test_same_seed_reproduces_split_and_other_seed_changes_it(
    silver_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    first = _make_builder(tmp_path, seed=42).transform(silver_sessions)
    repeated = _make_builder(tmp_path, seed=42).transform(silver_sessions)
    different = _make_builder(tmp_path, seed=7).transform(silver_sessions)

    pd.testing.assert_frame_equal(first.test, repeated.test)
    assert set(first.test[SESSION_ID]) != set(different.test[SESSION_ID])


def test_validate_output_rejects_overlapping_sessions(
    silver_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    builder = _make_builder(tmp_path)
    result = builder.transform(silver_sessions)
    leaked = GoldDatasets(
        train=result.train,
        test=pd.concat([result.test.iloc[1:], result.train.iloc[:1]]),
    )

    with pytest.raises(DataValidationError, match="treino e no teste"):
        builder.validate_output(silver_sessions, leaked)


def test_run_writes_train_and_test_files(
    silver_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    silver_sessions.to_parquet(tmp_path / "silver.parquet", index=False)

    _make_builder(tmp_path).run()

    train = pd.read_parquet(tmp_path / "gold" / "train.parquet")
    test = pd.read_parquet(tmp_path / "gold" / "test.parquet")
    assert len(train) + len(test) == len(silver_sessions)
