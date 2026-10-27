"""Inicialização compartilhada pelos estágios do pipeline."""

from purchase_intention.config import get_settings
from purchase_intention.logging_config import configure_logging
from purchase_intention.params import PipelineParams, load_params


def bootstrap() -> PipelineParams:
    """Configura o logging e carrega os parâmetros do pipeline.

    Returns:
        Parâmetros validados de ``configs/params.yaml``.
    """
    configure_logging(get_settings().log_level)
    return load_params()
