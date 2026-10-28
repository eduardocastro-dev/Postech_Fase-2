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

## Execução rápida (avaliadores)

Os dados já estão versionados no próprio repositório (remote DVC em `dvc-storage/`),
então não é preciso baixar nada nem configurar credenciais:

```bash
git clone https://github.com/eduardocastro-dev/Postech_Fase-2.git
cd Postech_Fase-2
poetry install
poetry run dvc repro --pull
```

O `dvc repro --pull` traz os dados e o modelo versionados, executa apenas os
estágios cujo código, dados ou parâmetros tenham mudado e publica o modelo no
MLflow local. Para ver os experimentos e o Model Registry no navegador:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
# abra http://127.0.0.1:5000
```

## Execução com Docker Compose (Fase 8)

Com Docker Engine e Docker Compose v2 instalados, execute na raiz do projeto:

```bash
docker compose up --build
```

O Compose constrói a imagem multi-stage usando o `poetry.lock`, inicia o
servidor MLflow e aguarda sua verificação de saúde antes de executar o pipeline
`dvc repro --pull`. O remote do DVC está incluído na imagem: não são
necessários `.env`, credenciais, instalação local de Python ou Poetry. Na
primeira execução, o pipeline recupera os artefatos versionados e publica o
modelo no MLflow. Nas seguintes, a publicação idempotente evita duplicatas.

A interface do MLflow fica em **http://127.0.0.1:5000**. O serviço `pipeline`
termina após a execução, enquanto `mlflow-server` continua atendendo. Para
consultar o resultado ou executar novamente:

```bash
docker compose logs pipeline
docker compose run --rm pipeline
docker compose down
```

O SQLite, os modelos registrados e os artefatos do MLflow persistem no volume
nomeado `mlflow-data`, mesmo após `docker compose down`. Para apagar
**definitivamente** esse estado de execução local, use `docker compose down -v`.
O servidor publica a porta apenas no localhost da máquina, não na rede externa.

## Pré-requisitos

- Python 3.11
- [Poetry](https://python-poetry.org/) 2.x (recomendado instalar via `pipx install poetry`)
- Git

## Instalação

```bash
poetry install
cp .env.example .env   # opcional: sem ele, valem os padrões de config.py
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

## Modelagem

| Etapa | Como é feita |
| --- | --- |
| Pré-processamento | `log1p` + `StandardScaler` em contagens e durações, `StandardScaler` nas taxas, `OneHotEncoder` nas categóricas. Tudo dentro do `Pipeline`, ajustado só nos dados de treino de cada dobra |
| Candidatos | `LogisticRegression`, `RandomForestClassifier` e `HistGradientBoostingClassifier`, todos com `class_weight="balanced"` |
| Seleção | Validação cruzada estratificada (5 dobras) no treino: maior F1 médio da classe positiva; a ROC-AUC desempata |
| Limiar de decisão | Escolhido para maximizar o F1 nas previsões *out-of-fold* do vencedor, sem usar o teste |
| Avaliação final | Uma única vez no conjunto de teste: F1, ROC-AUC, acurácia, precisão, recall e matriz de confusão |

Os hiperparâmetros ficam em `configs/params.yaml`. O modelo final é gravado em
`models/model.joblib`, com o limiar já embutido (`FixedThresholdClassifier`).
Os resultados ficam em `reports/cv_results.json` (seleção) e `reports/metrics.json`
(teste), e podem ser consultados com `poetry run dvc metrics show`.

## Rastreamento de experimentos (MLflow)

O treino acontece no DVC; o estágio `register` publica no MLflow o que o treino
produziu. Ele roda em todo `dvc repro`, mas é idempotente: o modelo é
identificado pelo SHA-256 do arquivo e, se já estiver publicado, nada é gravado.
Por isso um clone recém-feito preenche o MLflow local sem retreinar.

| Onde | O que fica registrado |
| --- | --- |
| Experimento `purchase-intention` | Uma execução `selection-<hash>` por modelo publicado, com seed, dobras, `test_size`, modelo escolhido, limiar e SHA-256 dos dados brutos; métricas `cv_*` do vencedor e `test_*` (incluindo a matriz de confusão); relatórios JSON e o modelo com assinatura e exemplo de entrada |
| Execuções filhas | Uma por candidato, com hiperparâmetros e média e desvio de cada métrica na validação cruzada |
| Model Registry `purchase-intention-classifier` | Uma versão por modelo publicado. O alias `champion` vai para a nova versão apenas se o F1 médio de validação dela superar o do campeão atual |

Para carregar o modelo campeão:

```python
import mlflow

model = mlflow.sklearn.load_model("models:/purchase-intention-classifier@champion")
```

O modelo é serializado em [skops](https://skops.readthedocs.io/), formato mais
seguro que o pickle. Os tipos internos que o skops pede para autorizar, como
as árvores do Random Forest, são liberados apenas para o modelo gerado pelo
próprio pipeline.

## Versionamento de dados (DVC)

O pipeline está declarado em `dvc.yaml`, e os hashes de cada entrada e saída
ficam em `dvc.lock`. Cada estágio declara o código e os parâmetros de que
depende. Por exemplo, alterar `split.test_size` refaz apenas o Gold.

| Comando | O que faz |
| --- | --- |
| `poetry run dvc repro` | Executa os estágios desatualizados |
| `poetry run dvc dag` | Mostra a ordem dos estágios |
| `poetry run dvc status` | Indica o que mudou desde a última execução (o `register` aparece sempre como alterado, por ser `always_changed`) |
| `poetry run dvc push` | Envia novos dados para o remote `dvc-storage/` |

O CSV bruto é rastreado por `data/raw/online_shoppers_intention.csv.dvc`.
Os arquivos de `data/` nunca entram no Git diretamente, apenas os ponteiros do DVC.

Cada camada também pode ser executada isoladamente, a partir da raiz do projeto:

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
