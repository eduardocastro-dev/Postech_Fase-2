"""Schema esperado do dataset e regras de validação compartilhadas pelas camadas.

As regras foram definidas a partir da análise exploratória
(``notebooks/01_analise_exploratoria.ipynb``). Se os dados mudarem e alguma
regra deixar de valer, o pipeline falha com uma mensagem clara em vez de
seguir com dados inconsistentes.
"""

from collections.abc import Collection, Iterable

import pandas as pd

TARGET = "Revenue"
SESSION_ID = "session_id"

NUMERIC_COLUMNS = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)
# Colunas inteiras em que cada número representa uma categoria, não uma quantidade.
CODE_COLUMNS = ("OperatingSystems", "Browser", "Region", "TrafficType")
CATEGORICAL_COLUMNS = ("Month", *CODE_COLUMNS, "VisitorType")
BOOLEAN_COLUMNS = ("Weekend", TARGET)

# Ordem original das colunas no CSV da UCI.
RAW_COLUMNS = (
    *NUMERIC_COLUMNS,
    "Month",
    *CODE_COLUMNS,
    "VisitorType",
    "Weekend",
    TARGET,
)

# Variáveis explicativas usadas pelo modelo (tudo, exceto alvo e identificador).
FEATURE_COLUMNS = (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, "Weekend")

# Colunas que representam proporções e, portanto, ficam entre 0 e 1.
RATE_COLUMNS = ("BounceRates", "ExitRates", "SpecialDay")

VALID_MONTHS = frozenset(
    {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    }
)  # fmt: skip
# Grafias fora do padrão encontradas no dataset bruto.
MONTH_ALIASES = {"June": "Jun"}
VALID_VISITOR_TYPES = frozenset({"Returning_Visitor", "New_Visitor", "Other"})


class DataValidationError(ValueError):
    """Erro lançado quando os dados violam alguma regra do schema."""


def validate_columns(data: pd.DataFrame, expected: Iterable[str]) -> None:
    """Garante que o DataFrame tenha exatamente as colunas esperadas.

    Raises:
        DataValidationError: Se faltar ou sobrar alguma coluna.
    """
    expected_set, actual_set = set(expected), set(data.columns)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        raise DataValidationError(
            f"Colunas divergentes. Faltando: {missing}. Inesperadas: {unexpected}."
        )


def validate_no_missing_values(data: pd.DataFrame) -> None:
    """Garante que não existam valores ausentes.

    Raises:
        DataValidationError: Se alguma coluna tiver valores ausentes.
    """
    missing_by_column = data.isna().sum()
    columns_with_missing = missing_by_column[missing_by_column > 0]
    if not columns_with_missing.empty:
        raise DataValidationError(
            f"Valores ausentes encontrados: {columns_with_missing.to_dict()}."
        )


def validate_range(
    data: pd.DataFrame,
    columns: Iterable[str],
    lower: float,
    upper: float | None = None,
) -> None:
    """Garante que os valores das colunas estejam no intervalo ``[lower, upper]``.

    Raises:
        DataValidationError: Se algum valor estiver fora do intervalo.
    """
    for column in columns:
        values = data[column]
        out_of_range = values < lower
        if upper is not None:
            out_of_range |= values > upper
        if out_of_range.any():
            raise DataValidationError(
                f"Coluna '{column}' tem {int(out_of_range.sum())} valores fora "
                f"do intervalo [{lower}, {upper}]."
            )


def validate_allowed_values(
    data: pd.DataFrame, column: str, allowed: Collection[str]
) -> None:
    """Garante que a coluna contenha apenas valores conhecidos.

    Raises:
        DataValidationError: Se aparecer algum valor fora da lista permitida.
    """
    unknown = sorted(set(data[column].unique()) - set(allowed))
    if unknown:
        raise DataValidationError(
            f"Coluna '{column}' tem valores desconhecidos: {unknown}."
        )


def validate_boolean_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    """Garante que as colunas tenham tipo booleano.

    Raises:
        DataValidationError: Se alguma coluna não for booleana.
    """
    not_boolean = [
        column for column in columns if not pd.api.types.is_bool_dtype(data[column])
    ]
    if not_boolean:
        raise DataValidationError(f"Colunas deveriam ser booleanas: {not_boolean}.")


def validate_row_count(actual: int, expected: int, context: str) -> None:
    """Garante que nenhuma linha foi perdida ou criada em uma transformação.

    Raises:
        DataValidationError: Se as contagens forem diferentes.
    """
    if actual != expected:
        raise DataValidationError(
            f"{context}: esperado {expected} linhas, obtido {actual}."
        )
