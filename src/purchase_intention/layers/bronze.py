"""Camada Bronze: ingestão do CSV bruto, sem alterar nenhum valor."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from purchase_intention.layers.base import LayerProcessor
from purchase_intention.layers.schema import (
    RAW_COLUMNS,
    DataValidationError,
    validate_columns,
    validate_row_count,
)
from purchase_intention.layers.storage import sha256_of_file, write_json, write_parquet


class BronzeIngestor(LayerProcessor[pd.DataFrame]):
    """Copia o CSV original para Parquet e registra os metadados da ingestão.

    A data e hora da ingestão fica apenas no JSON de metadados. Assim o Parquet
    sai idêntico a cada execução, e o DVC só detecta mudança quando o dado muda.
    """

    layer_name = "bronze"

    def __init__(
        self,
        source_path: Path,
        output_path: Path,
        metadata_path: Path,
        source_url: str,
    ) -> None:
        """Inicializa o ingestor.

        Args:
            source_path: CSV original baixado da UCI.
            output_path: Parquet de saída da camada Bronze.
            metadata_path: JSON com os metadados da ingestão.
            source_url: Endereço de onde o dataset foi obtido.
        """
        self._source_path = source_path
        self._output_path = output_path
        self._metadata_path = metadata_path
        self._source_url = source_url

    def read(self) -> pd.DataFrame:
        """Lê o CSV bruto com a inferência de tipos padrão do pandas."""
        return pd.read_csv(self._source_path)

    def validate_input(self, data: pd.DataFrame) -> None:
        """Confere se o CSV tem as colunas esperadas e não está vazio."""
        validate_columns(data, RAW_COLUMNS)
        if data.empty:
            raise DataValidationError("O arquivo bruto não tem nenhuma linha.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retorna os dados sem alteração: a Bronze preserva o conteúdo original."""
        return data

    def validate_output(self, data: pd.DataFrame, result: pd.DataFrame) -> None:
        """Garante que nenhuma linha ou coluna foi perdida."""
        validate_row_count(len(result), len(data), context="Bronze")
        validate_columns(result, RAW_COLUMNS)

    def write(self, result: pd.DataFrame) -> None:
        """Grava o Parquet e o JSON de metadados."""
        write_parquet(result, self._output_path)
        write_json(self._build_metadata(result), self._metadata_path)

    def _build_metadata(self, result: pd.DataFrame) -> dict[str, Any]:
        """Monta os metadados que garantem a rastreabilidade da origem."""
        return {
            "source_url": self._source_url,
            "source_file": self._source_path.as_posix(),
            "source_sha256": sha256_of_file(self._source_path),
            "rows": len(result),
            "columns": list(result.columns),
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
