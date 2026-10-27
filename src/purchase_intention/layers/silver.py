"""Camada Silver: validação de qualidade e padronização dos dados.

Aqui só acontecem transformações linha a linha. Nenhuma estatística é
aprendida (média, desvio, lista de categorias), o que evita vazamento de
informação entre treino e teste.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from purchase_intention.layers.base import LayerProcessor
from purchase_intention.layers.schema import (
    BOOLEAN_COLUMNS,
    CODE_COLUMNS,
    MONTH_ALIASES,
    NUMERIC_COLUMNS,
    RATE_COLUMNS,
    RAW_COLUMNS,
    SESSION_ID,
    TARGET,
    VALID_MONTHS,
    VALID_VISITOR_TYPES,
    DataValidationError,
    validate_allowed_values,
    validate_boolean_columns,
    validate_columns,
    validate_no_missing_values,
    validate_range,
    validate_row_count,
)
from purchase_intention.layers.storage import write_json, write_parquet

logger = logging.getLogger(__name__)

SILVER_COLUMNS = (SESSION_ID, *RAW_COLUMNS)


class SilverTransformer(LayerProcessor[pd.DataFrame]):
    """Valida a qualidade dos dados da Bronze e padroniza tipos e grafias."""

    layer_name = "silver"

    def __init__(self, input_path: Path, output_path: Path, report_path: Path) -> None:
        """Inicializa o transformador.

        Args:
            input_path: Parquet da camada Bronze.
            output_path: Parquet de saída da camada Silver.
            report_path: JSON com o relatório de qualidade.
        """
        self._input_path = input_path
        self._output_path = output_path
        self._report_path = report_path

    def read(self) -> pd.DataFrame:
        """Lê o Parquet da camada Bronze."""
        return pd.read_parquet(self._input_path)

    def validate_input(self, data: pd.DataFrame) -> None:
        """Aplica as regras de qualidade definidas na análise exploratória."""
        validate_columns(data, RAW_COLUMNS)
        validate_no_missing_values(data)
        validate_range(data, NUMERIC_COLUMNS, lower=0)
        validate_range(data, RATE_COLUMNS, lower=0, upper=1)
        validate_allowed_values(data, "Month", VALID_MONTHS | MONTH_ALIASES.keys())
        validate_allowed_values(data, "VisitorType", VALID_VISITOR_TYPES)
        validate_boolean_columns(data, BOOLEAN_COLUMNS)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Padroniza tipos e grafias, sem remover linhas nem aprender estatísticas."""
        result = data.copy()
        # Identificador estável: a posição da sessão no arquivo original.
        result.insert(0, SESSION_ID, range(len(result)))
        result["Month"] = result["Month"].replace(MONTH_ALIASES)
        # Códigos viram texto para nunca serem tratados como quantidades.
        for column in CODE_COLUMNS:
            result[column] = result[column].astype(str)
        return result

    def validate_output(self, data: pd.DataFrame, result: pd.DataFrame) -> None:
        """Garante que nenhuma linha se perdeu e que a padronização foi aplicada."""
        validate_row_count(len(result), len(data), context="Silver")
        validate_columns(result, SILVER_COLUMNS)
        validate_allowed_values(result, "Month", VALID_MONTHS)
        if not result[SESSION_ID].is_unique:
            raise DataValidationError(f"A coluna '{SESSION_ID}' tem valores repetidos.")

    def write(self, result: pd.DataFrame) -> None:
        """Grava o Parquet e o relatório de qualidade."""
        write_parquet(result, self._output_path)
        report = build_quality_report(result)
        logger.info(
            "[silver] %d linhas; %d duplicadas mantidas",
            report["rows"],
            report["duplicate_rows"],
        )
        write_json(report, self._report_path)


def build_quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """Resume a qualidade dos dados da Silver.

    As duplicatas são contadas, mas não removidas: a análise exploratória
    mostrou que são sessões legítimas de rejeição, idênticas por terem pouca
    informação.
    """
    duplicate_rows = int(data.drop(columns=SESSION_ID).duplicated().sum())
    return {
        "rows": len(data),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": round(duplicate_rows / len(data), 4),
        "duplicates_policy": "mantidas (ver notebooks/01_analise_exploratoria.ipynb)",
        "positive_rate": round(float(data[TARGET].mean()), 4),
    }
