"""Gravação e leitura do modelo treinado."""

from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin

# Compressão moderada: arquivo bem menor no remote do DVC, leitura ainda rápida.
_COMPRESSION_LEVEL = 3


def save_model(model: ClassifierMixin, path: Path) -> None:
    """Grava o modelo com joblib, criando a pasta se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=_COMPRESSION_LEVEL)


def load_model(path: Path) -> ClassifierMixin:
    """Lê um modelo gravado por ``save_model``.

    Atenção: joblib usa pickle; carregue apenas arquivos gerados por este projeto.
    """
    return joblib.load(path)
