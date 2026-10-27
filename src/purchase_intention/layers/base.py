"""Fluxo comum às camadas de dados (padrão Template Method)."""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd

OutputT = TypeVar("OutputT")

logger = logging.getLogger(__name__)


class LayerProcessor(ABC, Generic[OutputT]):
    """Define o fluxo ler → validar entrada → transformar → validar saída → gravar.

    A ordem das etapas é garantida aqui, em um único lugar. Cada camada
    implementa apenas o que muda, e a lógica de transformação fica separada
    da leitura e da escrita, o que facilita os testes.
    """

    layer_name: str = "camada"

    def run(self) -> OutputT:
        """Executa a camada de ponta a ponta e retorna o resultado gravado."""
        logger.info("[%s] Iniciando processamento", self.layer_name)
        data = self.read()
        self.validate_input(data)
        result = self.transform(data)
        self.validate_output(data, result)
        self.write(result)
        logger.info("[%s] Processamento concluído", self.layer_name)
        return result

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Lê os dados de entrada da camada."""

    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> None:
        """Verifica se a entrada respeita o schema esperado."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> OutputT:
        """Aplica as transformações da camada."""

    @abstractmethod
    def validate_output(self, data: pd.DataFrame, result: OutputT) -> None:
        """Verifica se a saída é consistente com a entrada."""

    @abstractmethod
    def write(self, result: OutputT) -> None:
        """Grava o resultado da camada."""
