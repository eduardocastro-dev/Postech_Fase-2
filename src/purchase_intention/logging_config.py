"""Configuração padronizada de logging para todos os estágios do pipeline."""

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configura o logger raiz com formato e nível padronizados.

    Args:
        level: Nível mínimo das mensagens exibidas (ex.: ``"INFO"``).
    """
    # force=True substitui configurações anteriores, evitando logs duplicados
    # quando mais de um estágio é executado no mesmo processo.
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
