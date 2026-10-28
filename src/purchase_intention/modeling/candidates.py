"""Catálogo dos modelos candidatos, montados a partir do ``params.yaml``."""

from collections.abc import Mapping
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from purchase_intention.modeling.preprocessing import build_preprocessor

# Para incluir um novo candidato, basta registrá-lo aqui e no params.yaml.
ESTIMATORS: dict[str, type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def build_candidate(
    name: str, hyperparameters: Mapping[str, Any], seed: int
) -> Pipeline:
    """Monta o Pipeline (pré-processamento + classificador) de um candidato.

    Todos usam ``class_weight="balanced"`` para compensar a classe positiva
    minoritária e a mesma semente, para que os resultados sejam reprodutíveis.

    Raises:
        ValueError: Se o nome não estiver no catálogo.
    """
    if name not in ESTIMATORS:
        raise ValueError(
            f"Modelo desconhecido: '{name}'. Disponíveis: {sorted(ESTIMATORS)}."
        )
    classifier = ESTIMATORS[name](
        **hyperparameters, class_weight="balanced", random_state=seed
    )
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("classifier", classifier)]
    )


def build_candidates(
    models: Mapping[str, Mapping[str, Any]], seed: int
) -> dict[str, Pipeline]:
    """Monta todos os candidatos declarados no ``params.yaml``."""
    return {
        name: build_candidate(name, hyperparameters, seed)
        for name, hyperparameters in models.items()
    }
