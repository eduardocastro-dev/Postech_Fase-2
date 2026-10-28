# Estágio de construção: dependências de produção fixadas pelo poetry.lock.
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv \
    && /usr/local/bin/python -m pip install --no-cache-dir "poetry==2.2.1"

WORKDIR /app

# Instala dependências antes do código para aproveitar o cache do Docker.
COPY pyproject.toml poetry.lock poetry.toml README.md ./
RUN poetry install --only main --no-root --sync --no-interaction --no-ansi

COPY src/ ./src/
RUN poetry install --only-root --no-interaction --no-ansi

# Imagem final: sem Poetry e sem ferramentas de desenvolvimento.
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DVC_NO_ANALYTICS=true

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app /var/lib/mlflow/artifacts \
    && chown -R app:app /app /var/lib/mlflow

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Mantém a estrutura esperada pelos estágios e pelo remote local do DVC.
COPY --chown=app:app .dvc/ ./.dvc/
COPY --chown=app:app .dvcignore dvc.yaml dvc.lock ./
COPY --chown=app:app dvc-storage/ ./dvc-storage/
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app src/ ./src/

USER app

CMD ["dvc", "repro", "--pull"]
