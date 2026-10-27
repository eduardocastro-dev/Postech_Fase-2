import json
from pathlib import Path

import pandas as pd
import pytest

from purchase_intention.layers.schema import (
    CODE_COLUMNS,
    SESSION_ID,
    DataValidationError,
)
from purchase_intention.layers.silver import SilverTransformer


@pytest.fixture
def transformer(tmp_path: Path) -> SilverTransformer:
    return SilverTransformer(
        input_path=tmp_path / "bronze.parquet",
        output_path=tmp_path / "silver" / "sessions.parquet",
        report_path=tmp_path / "silver" / "quality_report.json",
    )


def test_standardizes_june_to_three_letters(
    transformer: SilverTransformer, raw_sessions: pd.DataFrame
) -> None:
    result = transformer.transform(raw_sessions)

    assert "June" not in set(result["Month"])
    assert result.loc[0, "Month"] == "Jun"


def test_code_columns_become_text(
    transformer: SilverTransformer, raw_sessions: pd.DataFrame
) -> None:
    result = transformer.transform(raw_sessions)

    for column in CODE_COLUMNS:
        assert pd.api.types.is_string_dtype(result[column]), column


def test_adds_session_id_without_losing_rows(
    transformer: SilverTransformer, raw_sessions: pd.DataFrame
) -> None:
    result = transformer.transform(raw_sessions)

    assert len(result) == len(raw_sessions)
    assert result[SESSION_ID].tolist() == list(range(len(raw_sessions)))


def test_transform_does_not_modify_input(
    transformer: SilverTransformer, raw_sessions: pd.DataFrame
) -> None:
    original = raw_sessions.copy()

    transformer.transform(raw_sessions)

    pd.testing.assert_frame_equal(raw_sessions, original)


def test_run_keeps_duplicates_and_reports_them(
    transformer: SilverTransformer, raw_sessions: pd.DataFrame, tmp_path: Path
) -> None:
    raw_sessions.to_parquet(tmp_path / "bronze.parquet", index=False)

    transformer.run()

    silver = pd.read_parquet(tmp_path / "silver" / "sessions.parquet")
    report_path = tmp_path / "silver" / "quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(silver) == len(raw_sessions)
    assert report["duplicate_rows"] == 1
    assert report["positive_rate"] == 0.2


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("PageValues", None, "ausentes"),
        ("ProductRelated_Duration", -1.0, "ProductRelated_Duration"),
        ("BounceRates", 1.5, "BounceRates"),
        ("Month", "Janeiro", "Janeiro"),
        ("VisitorType", "Bot", "Bot"),
    ],
)
def test_rejects_data_that_violates_schema(
    transformer: SilverTransformer,
    raw_sessions: pd.DataFrame,
    column: str,
    invalid_value: object,
    message: str,
) -> None:
    raw_sessions[column] = raw_sessions[column].astype(object)
    raw_sessions.loc[5, column] = invalid_value

    with pytest.raises(DataValidationError, match=message):
        transformer.validate_input(raw_sessions.infer_objects())


def test_rejects_target_that_is_not_boolean(
    transformer: SilverTransformer, raw_sessions: pd.DataFrame
) -> None:
    raw_sessions["Revenue"] = raw_sessions["Revenue"].map({True: "yes", False: "no"})

    with pytest.raises(DataValidationError, match="Revenue"):
        transformer.validate_input(raw_sessions)
