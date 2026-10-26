"""Configurações de infraestrutura lidas das variáveis de ambiente e do arquivo .env.

Os parâmetros de experimento (seeds, tamanho do split, hiperparâmetros) ficam
de fora deste módulo de propósito: eles vivem em ``configs/params.yaml`` para
que o DVC consiga rastreá-los e reexecutar o pipeline quando mudarem.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Configurações que mudam entre ambientes (máquina local, Docker).

    Ordem de prioridade: variável de ambiente > arquivo ``.env`` > valor padrão.
    Cada campo é lido da variável de ambiente com o mesmo nome em maiúsculas.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLFLOW_TRACKING_URI e MLFLOW_EXPERIMENT_NAME são nomes oficiais do MLflow,
    # que também os lê diretamente: a mesma configuração vale para o nosso
    # código e para a própria biblioteca.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "purchase-intention"
    mlflow_registered_model_name: str = "purchase-intention-classifier"
    log_level: LogLevel = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Retorna as configurações da aplicação, carregadas uma única vez por processo."""
    return Settings()
