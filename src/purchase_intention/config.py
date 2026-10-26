"""Configurações de infraestrutura lidas das variáveis de ambiente e do arquivo .env.

Os parâmetros de experimento (seeds, tamanho do split, hiperparâmetros) ficam
de fora deste módulo de propósito: eles vivem em ``configs/params.yaml`` para
que o DVC consiga rastreá-los e reexecutar o pipeline quando mudarem.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

NivelLog = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Configuracoes(BaseSettings):
    """Configurações que mudam entre ambientes (máquina local, Docker).

    Ordem de prioridade: variável de ambiente > arquivo ``.env`` > valor padrão.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # As duas primeiras variáveis mantêm o nome oficial do MLflow, que também
    # as lê diretamente. Assim a mesma configuração vale para o nosso código
    # e para a própria biblioteca.
    uri_rastreamento_mlflow: str = Field(
        default="sqlite:///mlflow.db",
        validation_alias="MLFLOW_TRACKING_URI",
    )
    nome_experimento: str = Field(
        default="intencao-de-compra",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )
    nome_modelo_registrado: str = Field(
        default="classificador-intencao-de-compra",
        validation_alias="NOME_MODELO_REGISTRADO",
    )
    nivel_log: NivelLog = Field(default="INFO", validation_alias="NIVEL_LOG")


@lru_cache
def obter_configuracoes() -> Configuracoes:
    """Retorna as configurações da aplicação, carregadas uma única vez por processo."""
    return Configuracoes()
