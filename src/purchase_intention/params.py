"""Leitura e validação dos parâmetros do pipeline (``configs/params.yaml``).

Os modelos Pydantic garantem que um erro de digitação ou um valor inválido no
YAML falhe logo no início da execução, com uma mensagem clara.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PARAMS_PATH = Path("configs/params.yaml")


class _StrictModel(BaseModel):
    """Base que rejeita chaves desconhecidas e impede alterações após a leitura."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataParams(_StrictModel):
    """Origem do dataset e caminhos de entrada e saída de cada camada."""

    source_url: str
    raw: Path
    bronze: Path
    bronze_metadata: Path
    silver: Path
    silver_report: Path
    gold_train: Path
    gold_test: Path


class SplitParams(_StrictModel):
    """Parâmetros da divisão entre treino e teste."""

    test_size: float = Field(gt=0, lt=1)


class PipelineParams(_StrictModel):
    """Conjunto completo de parâmetros do pipeline."""

    seed: int
    data: DataParams
    split: SplitParams


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> PipelineParams:
    """Lê o arquivo YAML de parâmetros e valida seu conteúdo.

    Args:
        path: Caminho do arquivo de parâmetros.

    Returns:
        Parâmetros validados e imutáveis.
    """
    with path.open(encoding="utf-8") as params_file:
        raw_params = yaml.safe_load(params_file)
    return PipelineParams.model_validate(raw_params)
