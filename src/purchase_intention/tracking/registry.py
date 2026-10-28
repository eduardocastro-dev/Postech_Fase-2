"""Registro de versões no MLflow Model Registry e promoção do campeão.

O MLflow 3 usa aliases no lugar dos antigos *stages*. A versão com o alias
``champion`` é a que deve ser usada para previsões.
"""

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION_ALIAS = "champion"
# Tag da versão com o critério de seleção, usada para comparar com o campeão.
CV_F1_TAG = "cv_f1"

logger = logging.getLogger(__name__)


def register_version(model_uri: str, name: str, cv_f1: float) -> str:
    """Registra uma nova versão do modelo e devolve o número dela."""
    version = mlflow.register_model(model_uri, name, tags={CV_F1_TAG: str(cv_f1)})
    # A API documenta a versão como texto, mas o store SQLite devolve int.
    return str(version.version)


def champion_cv_f1(client: MlflowClient, name: str) -> float | None:
    """F1 médio de validação do campeão atual, ou ``None`` se não houver campeão."""
    try:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return float(champion.tags[CV_F1_TAG])


def promote_if_better(
    client: MlflowClient, name: str, version: str, cv_f1: float
) -> bool:
    """Dá o alias ``champion`` à versão se ela superar o campeão atual.

    O critério é o mesmo da seleção: F1 médio na validação cruzada. O conjunto
    de teste não é usado para decidir.

    Returns:
        ``True`` se a versão virou campeã.
    """
    current = champion_cv_f1(client, name)
    if current is not None and cv_f1 <= current:
        logger.info(
            "[register] Versão %s (F1 %.4f) não supera o campeão (F1 %.4f)",
            version,
            cv_f1,
            current,
        )
        return False
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    return True
