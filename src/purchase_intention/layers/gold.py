"""Camada Gold: conjuntos de treino e teste prontos para a modelagem.

O split acontece aqui, antes de qualquer transformação que aprenda com os
dados. Escalonamento e codificação ficam dentro do Pipeline do scikit-learn,
ajustados apenas no conjunto de treino.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_intention.layers.base import LayerProcessor
from purchase_intention.layers.schema import (
    FEATURE_COLUMNS,
    SESSION_ID,
    TARGET,
    DataValidationError,
    validate_columns,
    validate_no_missing_values,
    validate_row_count,
)
from purchase_intention.layers.storage import write_parquet

GOLD_COLUMNS = (SESSION_ID, *FEATURE_COLUMNS, TARGET)


@dataclass(frozen=True)
class GoldDatasets:
    """Conjuntos de treino e teste produzidos pela camada Gold."""

    train: pd.DataFrame
    test: pd.DataFrame


class GoldBuilder(LayerProcessor[GoldDatasets]):
    """Converte o alvo para 0/1 e separa treino e teste de forma estratificada."""

    layer_name = "gold"

    def __init__(
        self,
        input_path: Path,
        train_path: Path,
        test_path: Path,
        test_size: float,
        seed: int,
        max_positive_rate_gap: float = 0.01,
    ) -> None:
        """Inicializa o construtor.

        Args:
            input_path: Parquet da camada Silver.
            train_path: Parquet de saída com o conjunto de treino.
            test_path: Parquet de saída com o conjunto de teste.
            test_size: Proporção das sessões reservada para teste.
            seed: Semente que torna o split reprodutível.
            max_positive_rate_gap: Diferença máxima aceita entre a proporção
                de compras no treino e no teste.
        """
        self._input_path = input_path
        self._train_path = train_path
        self._test_path = test_path
        self._test_size = test_size
        self._seed = seed
        self._max_positive_rate_gap = max_positive_rate_gap

    def read(self) -> pd.DataFrame:
        """Lê o Parquet da camada Silver."""
        return pd.read_parquet(self._input_path)

    def validate_input(self, data: pd.DataFrame) -> None:
        """Confere se a entrada tem as colunas da Silver e nenhum valor ausente."""
        validate_columns(data, GOLD_COLUMNS)
        validate_no_missing_values(data)

    def transform(self, data: pd.DataFrame) -> GoldDatasets:
        """Converte o alvo para 0/1 e faz o split estratificado com semente fixa."""
        prepared = data.loc[:, list(GOLD_COLUMNS)].astype({TARGET: int})
        train, test = train_test_split(
            prepared,
            test_size=self._test_size,
            random_state=self._seed,
            stratify=prepared[TARGET],
        )
        return GoldDatasets(
            train=train.reset_index(drop=True),
            test=test.reset_index(drop=True),
        )

    def validate_output(self, data: pd.DataFrame, result: GoldDatasets) -> None:
        """Garante que o split não perdeu, duplicou nem desbalanceou sessões."""
        total = len(result.train) + len(result.test)
        validate_row_count(total, len(data), context="Gold")
        self._validate_no_overlap(result)
        self._validate_stratification(result)

    def write(self, result: GoldDatasets) -> None:
        """Grava os conjuntos de treino e teste."""
        write_parquet(result.train, self._train_path)
        write_parquet(result.test, self._test_path)

    @staticmethod
    def _validate_no_overlap(result: GoldDatasets) -> None:
        """Garante que nenhuma sessão aparece no treino e no teste ao mesmo tempo."""
        shared = set(result.train[SESSION_ID]) & set(result.test[SESSION_ID])
        if shared:
            raise DataValidationError(
                f"{len(shared)} sessões aparecem no treino e no teste."
            )

    def _validate_stratification(self, result: GoldDatasets) -> None:
        """Garante as duas classes em cada conjunto e proporções parecidas."""
        for name, subset in (("treino", result.train), ("teste", result.test)):
            if subset[TARGET].nunique() != 2:
                raise DataValidationError(f"O conjunto de {name} não tem as 2 classes.")
        gap = abs(result.train[TARGET].mean() - result.test[TARGET].mean())
        if gap > self._max_positive_rate_gap:
            raise DataValidationError(
                f"Proporção de compras difere {gap:.2%} entre treino e teste."
            )
