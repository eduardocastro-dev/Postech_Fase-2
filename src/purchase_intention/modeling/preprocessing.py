"""Pré-processamento das variáveis, aplicado dentro do Pipeline do modelo.

Por ficar dentro do Pipeline, cada transformação aprende apenas com os dados
de treino de cada dobra da validação cruzada, sem vazamento para a validação
nem para o teste.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from purchase_intention.layers.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RATE_COLUMNS,
    TARGET,
)

# Contagens, durações e PageValues: muito assimétricas, com caudas longas (ver EDA).
SKEWED_COLUMNS = tuple(
    column for column in NUMERIC_COLUMNS if column not in RATE_COLUMNS
)
BOOLEAN_FEATURES = ("Weekend",)


def build_preprocessor() -> ColumnTransformer:
    """Monta o transformador de colunas usado por todos os modelos candidatos.

    Returns:
        ``ColumnTransformer`` ainda não ajustado.
    """
    skewed = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    # Saída densa: o HistGradientBoosting não aceita matrizes esparsas.
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("skewed", skewed, list(SKEWED_COLUMNS)),
            ("rates", StandardScaler(), list(RATE_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
            ("boolean", "passthrough", list(BOOLEAN_FEATURES)),
        ]
    )


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as variáveis explicativas do alvo, descartando o identificador.

    Returns:
        Tupla ``(features, target)``.
    """
    return data.loc[:, list(FEATURE_COLUMNS)], data[TARGET]
