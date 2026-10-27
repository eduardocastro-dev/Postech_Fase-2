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

## Arquitetura de dados (medalhão)

| Camada | Classe | O que faz | Saída |
| --- | --- | --- | --- |
| Bronze | `BronzeIngestor` | Copia o CSV original para Parquet, sem alterar valores, e registra origem e SHA-256 | `data/bronze/sessions.parquet`, `metadata.json` |
| Silver | `SilverTransformer` | Valida o schema, padroniza `Month` e converte códigos em texto; conta duplicatas sem removê-las | `data/silver/sessions.parquet`, `quality_report.json` |
| Gold | `GoldBuilder` | Converte o alvo para 0/1 e faz o split estratificado (80/20, seed 42) | `data/gold/train.parquet`, `test.parquet` |

As três camadas seguem o mesmo fluxo, definido uma única vez em `LayerProcessor`:
ler → validar entrada → transformar → validar saída → gravar. Nenhuma estatística é
aprendida antes do split; escalonamento e codificação ficam no Pipeline do modelo.

Execução manual, a partir da raiz do projeto:

```bash
poetry run python -m purchase_intention.stages.ingest_bronze
poetry run python -m purchase_intention.stages.transform_silver
poetry run python -m purchase_intention.stages.build_gold
```

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

As variáveis `MLFLOW_TRACKING_URI` e `MLFLOW_EXPERIMENT_NAME` são nomes
oficiais do MLflow, que também as lê diretamente.

## Convenções do código

| Elemento | Idioma |
| --- | --- |
| Pastas, arquivos, variáveis, funções, classes | Inglês |
| Comentários e docstrings | Português |
