from pathlib import Path

import pytest
from pydantic import ValidationError

from purchase_intention.config import Configuracoes

VARIAVEIS_DE_AMBIENTE = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "NOME_MODELO_REGISTRADO",
    "NIVEL_LOG",
)


@pytest.fixture(autouse=True)
def limpar_ambiente(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remove variáveis que possam existir na máquina, para que cada teste
    # parta de um ambiente conhecido.
    for nome in VARIAVEIS_DE_AMBIENTE:
        monkeypatch.delenv(nome, raising=False)


def test_usa_valores_padrao_sem_arquivo_env() -> None:
    # _env_file=None ignora um eventual .env local do desenvolvedor.
    configuracoes = Configuracoes(_env_file=None)

    assert configuracoes.uri_rastreamento_mlflow == "sqlite:///mlflow.db"
    assert configuracoes.nome_experimento == "intencao-de-compra"
    assert configuracoes.nome_modelo_registrado == "classificador-intencao-de-compra"
    assert configuracoes.nivel_log == "INFO"


def test_variaveis_de_ambiente_sobrescrevem_padroes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    monkeypatch.setenv("NIVEL_LOG", "DEBUG")

    configuracoes = Configuracoes(_env_file=None)

    assert configuracoes.uri_rastreamento_mlflow == "http://mlflow-server:5000"
    assert configuracoes.nivel_log == "DEBUG"


def test_carrega_valores_do_arquivo_env(tmp_path: Path) -> None:
    arquivo_env = tmp_path / ".env"
    arquivo_env.write_text("MLFLOW_EXPERIMENT_NAME=vindo-do-env\n", encoding="utf-8")

    configuracoes = Configuracoes(_env_file=arquivo_env)

    assert configuracoes.nome_experimento == "vindo-do-env"


def test_rejeita_nivel_de_log_invalido(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIVEL_LOG", "DETALHADO")

    with pytest.raises(ValidationError):
        Configuracoes(_env_file=None)
