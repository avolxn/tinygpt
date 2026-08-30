FROM apache/airflow:3.1.7-python3.12

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.10.9/install.sh -o /tmp/uv-install.sh \
    && env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh \
    && rm /tmp/uv-install.sh

ARG TINYGPT_EXTRA=cpu
ENV UV_PROJECT_ENVIRONMENT=/opt/tinygpt/.venv \
    UV_LINK_MODE=copy

WORKDIR /opt/tinygpt
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra "${TINYGPT_EXTRA}" --extra spark --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY tasks ./tasks
RUN uv sync --frozen --no-dev --extra "${TINYGPT_EXTRA}" --extra spark

COPY infra/airflow/dags /opt/airflow/dags

USER airflow
WORKDIR /workspace
