# Purchase Intention Prediction

Tech Challenge — Fase 2 · FIAP Pós Tech (Machine Learning Engineering)

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

## Resultados do modelo

O conjunto original contém **12.330 sessões**, das quais **15,47%** terminaram
em compra. A divisão estratificada (seed 42) reservou **9.864 sessões para
seleção/treinamento** e **2.466 para avaliação final**. Os dados do teste não
participaram da escolha do modelo nem do ajuste do limiar de classificação.

Três classificadores com pesos de classe balanceados foram comparados por
**validação cruzada estratificada de cinco dobras**, exclusivamente no treino.
O critério principal foi o F1 médio da classe positiva (`Revenue = 1`).

| Modelo candidato | F1 médio (± desvio) | ROC-AUC média | Precisão média | Recall médio |
| --- | ---: | ---: | ---: | ---: |
| **Random Forest (selecionado)** | **0,6789 ± 0,0164** | **0,9303** | 0,5961 | 0,7896 |
| HistGradientBoosting | 0,6697 ± 0,0132 | 0,9274 | 0,6038 | 0,7529 |
| Regressão Logística | 0,6558 ± 0,0175 | 0,9197 | 0,5553 | 0,8014 |

O limiar foi otimizado com previsões *out-of-fold* do vencedor e fixado em
**0,586776**, sem consulta ao conjunto de teste.

**Avaliação final no teste (2.466 sessões):**

| Métrica | Valor |
| --- | ---: |
| **F1 da classe compra** | **0,6617** |
| **ROC-AUC** | **0,9236** |
| Acurácia | 0,8905 |
| Precisão (compra) | 0,6346 |
| Recall (compra) | 0,6911 |

Matriz de confusão (linhas = classe real; colunas = predição):

| Classe real | Previsto sem compra | Previsto compra |
| --- | ---: | ---: |
| Sem compra (`Revenue = 0`) | 1.932 (VN) | 152 (FP) |
| Compra (`Revenue = 1`) | 118 (FN) | 264 (VP) |

**Fonte dos resultados:** `reports/cv_results.json` e `reports/metrics.json`,
ambos versionados no DVC e referenciados por `dvc.lock`. Os valores foram
extraídos dos artefatos deste release, não de uma execução hipotética.
O notebook exploratório documenta as decisões de preparação dos dados.

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

## Validação da entrega em clone limpo (Fase 9)

Após enviar o commit final para o GitHub, execute **na pasta do projeto local**:

```bash
python scripts/validate_clean_clone.py
```

Requisitos: acesso à internet para clonar, Git, Python 3.11 e Poetry 2.x
instalados no computador que inicia a validação. O script usa **um diretório
temporário independente**, sem reaproveitar `.venv`, `.env`, `mlflow.db`, dados
ou modelos da sua cópia de trabalho. Ele:

1. Clona a branch `main` diretamente do GitHub.
2. Instala as dependências do `poetry.lock` e executa Ruff e pytest.
3. Executa `dvc repro --pull`, recuperando os artefatos do remote local.
4. Confere o SHA-256 do CSV bruto e os resultados versionados no DVC.
5. Executa o DVC novamente e verifica que o Registry tem **uma versão** com
   alias `champion`, cujas previsões coincidem com as do modelo do DVC.
6. Exibe as métricas do DVC e remove o clone temporário ao concluir.

Para verificar um **commit local antes do push**, execute:

```bash
python scripts/validate_clean_clone.py --repository .
```

Para inspecionar a pasta gerada, acrescente `--keep`. Também é possível usar
`--repository CAMINHO_OU_URL` e `--branch NOME` para validar outro repositório
ou branch. **O script valida apenas o que estiver commitado na origem**:
modificações não commitadas não entram no clone, e o comando padrão valida
o que já estiver publicado no GitHub.

Depois de uma publicação bem-sucedida, `register` pode aparecer como alterado
em `dvc status`: isso é esperado por causa de `always_changed`. O mecanismo de
idempotência evita adicionar versões duplicadas ao Registry.

A validação com Docker depende do Docker Engine instalado no computador:

```bash
docker compose config
docker compose up --build
# Em outro terminal:
docker compose logs pipeline
```

A interface deve estar disponível em `http://127.0.0.1:5000`. Para encerrar:
`docker compose down` (mantém o volume do MLflow).

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

Os dois serviços usam a mesma imagem, construída uma única vez pelo
`mlflow-server`; o `pipeline` apenas a reutiliza (`pull_policy: never`). Por
isso o `docker compose run --rm pipeline` exige que a imagem já tenha sido
construída pelo `docker compose up --build`. A pasta `.git` não entra na
imagem, e o DVC é configurado no build para trabalhar sem Git
(`core.no_scm`).

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
depende. Por exemplo, alterar `split.test_size` refaz a camada Gold e os estágios
posteriores, sem refazer Bronze ou Silver.

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

## Limitações e decisões técnicas

- **Possível vazamento temporal em `PageValues`:** o indicador do Google
  Analytics tem forte relação com conversões e pode incorporar informação
  associada a compras. Ele foi mantido conforme o dataset e o estudo original,
  mas a aplicação em tempo real exige comprovar que seu valor esteja disponível
  **antes** da decisão de compra. Sem essa garantia, as métricas podem ser
  otimistas; recomenda-se avaliar um modelo sem essa variável.
- **Desbalanceamento:** apenas 15,47% das sessões são positivas. Por isso
  utilizamos split estratificado, `class_weight="balanced"`, F1 da classe
  positiva e ajuste do limiar. Precisão e recall dependem da taxa de compra
  observada e podem mudar em outros períodos ou lojas.
- **Duplicatas preservadas:** as **125 linhas duplicadas (1,01%)** foram
  mantidas por representarem padrões de sessões sem compra. Registros com
  características idênticas podem aparecer nos dois conjuntos, o que limita
  a interpretação da avaliação como generalização para usuários inéditos.
- **Cobertura incompleta das categorias:** janeiro e abril não aparecem nos
  dados disponíveis. `OneHotEncoder(handle_unknown="ignore")` permite receber
  categorias desconhecidas, mas não fornece evidência de desempenho para elas.
- **Treino e validação:** hiperparâmetros fixos, sem busca automatizada;
  avaliação em um único split estratificado, sem teste temporal ou validação
  externa. O modelo não foi validado em tráfego de produção.
- **Infraestrutura de demonstração:** o remote DVC fica no próprio repositório
  para dispensar credenciais, e o MLflow do Compose usa SQLite em volume
  persistente e acesso localhost. Em produção, seriam recomendados storage
  externo, autenticação, observabilidade e políticas de retenção.
- **Escopo da entrega:** não inclui deploy em nuvem nem Kubernetes, que não
  são requisitos obrigatórios do Tech Challenge.

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
