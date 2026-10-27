"""Leitura e escrita de arquivos das camadas, sempre criando as pastas necessárias."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

_HASH_CHUNK_SIZE = 1024 * 1024  # lê o arquivo em blocos de 1 MB


def write_parquet(data: pd.DataFrame, path: Path) -> None:
    """Grava o DataFrame em Parquet, sem o índice do pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Grava um dicionário em JSON legível, com acentos preservados."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(content + "\n", encoding="utf-8")


def sha256_of_file(path: Path) -> str:
    """Calcula o SHA-256 do arquivo, usado como prova de origem dos dados."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
