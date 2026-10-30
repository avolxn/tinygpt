FROM apache/airflow:3.1.7-python3.12

COPY --from=docker:28-cli /usr/local/bin/docker /usr/local/bin/docker

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl openssh-client \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.10.9/install.sh -o /tmp/uv-install.sh \
    && env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh \
    && rm /tmp/uv-install.sh

ARG TINYGPT_EXTRA=cpu
ENV UV_PROJECT_ENVIRONMENT=/opt/tinygpt/.venv \
    UV_LINK_MODE=copy

WORKDIR /opt/tinygpt
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra "${TINYGPT_EXTRA}" --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY tasks ./tasks
RUN uv sync --frozen --no-dev --extra "${TINYGPT_EXTRA}"

COPY infra/airflow/dags /opt/airflow/dags
RUN mkdir -p /mlflow/artifacts && chown -R airflow:root /mlflow

USER airflow
WORKDIR /workspace
