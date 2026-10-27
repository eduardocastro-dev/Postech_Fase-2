import pandas as pd
import pytest

from purchase_intention.layers.schema import (
    DataValidationError,
    validate_allowed_values,
    validate_boolean_columns,
    validate_columns,
    validate_no_missing_values,
    validate_range,
    validate_row_count,
)


@pytest.fixture
def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {"rate": [0.0, 0.5, 1.0], "month": ["Feb", "Mar", "Nov"], "flag": [True] * 3}
    )


def test_validate_columns_accepts_same_columns_in_any_order(
    sample: pd.DataFrame,
) -> None:
    validate_columns(sample, ["flag", "month", "rate"])


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        (["rate", "month", "flag", "extra"], "Faltando: \\['extra'\\]"),
        (["rate", "month"], "Inesperadas: \\['flag'\\]"),
    ],
)
def test_validate_columns_reports_divergences(
    sample: pd.DataFrame, expected: list[str], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_columns(sample, expected)


def test_validate_no_missing_values_rejects_nan(sample: pd.DataFrame) -> None:
    sample.loc[1, "rate"] = None

    with pytest.raises(DataValidationError, match="rate"):
        validate_no_missing_values(sample)


@pytest.mark.parametrize("invalid_value", [-0.1, 1.1])
def test_validate_range_rejects_values_outside_interval(
    sample: pd.DataFrame, invalid_value: float
) -> None:
    sample.loc[0, "rate"] = invalid_value

    with pytest.raises(DataValidationError, match="rate"):
        validate_range(sample, ["rate"], lower=0, upper=1)


def test_validate_range_without_upper_bound_accepts_large_values(
    sample: pd.DataFrame,
) -> None:
    sample.loc[0, "rate"] = 1_000.0

    validate_range(sample, ["rate"], lower=0)


def test_validate_allowed_values_lists_unknown_values(sample: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Nov"):
        validate_allowed_values(sample, "month", {"Feb", "Mar"})


def test_validate_boolean_columns_rejects_text(sample: pd.DataFrame) -> None:
    sample["flag"] = ["TRUE", "FALSE", "TRUE"]

    with pytest.raises(DataValidationError, match="flag"):
        validate_boolean_columns(sample, ["flag"])


def test_validate_row_count_rejects_lost_rows() -> None:
    with pytest.raises(DataValidationError, match="esperado 10 linhas, obtido 9"):
        validate_row_count(9, 10, context="Teste")
