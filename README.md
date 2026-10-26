# Purchase Intention Prediction

Tech Challenge — Fase 2 · FIAP Pós Tech (Machine Learning Engineering)

> Documento em construção: será completado ao longo das etapas do projeto.

## Contexto

Uma empresa de e-commerce precisa identificar a **propensão de compra** de seus
usuários a partir do comportamento de navegação. O foco do projeto é a
**Engenharia de Machine Learning**: reprodutibilidade, versionamento de dados,
rastreabilidade de experimentos e código limpo.

**Dataset:** [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(UCI) — classificação binária, alvo `Revenue`.

| Item | Valor |
| --- | --- |
| Arquivo | `data/raw/online_shoppers_intention.csv` |
| Origem | [zip oficial da UCI](https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip) |
| Dimensões | 12.330 sessões × 18 colunas |
| Classe positiva (`Revenue = True`) | 15,5% |
| SHA-256 | `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` |

A análise exploratória e as decisões de preparação dos dados estão em
[`notebooks/01_analise_exploratoria.ipynb`](notebooks/01_analise_exploratoria.ipynb).

## Pré-requisitos

- Python 3.11
- [Poetry](https://python-poetry.org/) 2.x (recomendado instalar via `pipx install poetry`)
- Git

## Instalação

```bash
poetry install
cp .env.example .env   # ajuste os valores se necessário
```

O arquivo `poetry.toml` configura o ambiente virtual dentro do projeto (`.venv/`).

## Qualidade de código

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run pytest
```

## Configuração

| Onde | O quê |
| --- | --- |
| `.env` | Infraestrutura que varia por ambiente (URI do MLflow, nomes de experimento e modelo, nível de log) |
| `configs/params.yaml` | Parâmetros de experimento (seeds, split, hiperparâmetros), rastreados pelo DVC |

As variáveis `MLFLOW_TRACKING_URI` e `MLFLOW_EXPERIMENT_NAME` mantêm o nome
oficial do MLflow, que também as lê diretamente.
