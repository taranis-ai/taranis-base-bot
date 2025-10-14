FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app/

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-dev \
    git

COPY . /app/

ENV UV_COMPILE_BYTECODE=1

RUN uv venv && \
    export PATH="/app/.venv/bin:$PATH" && \
    uv sync --frozen

FROM python:3.13-slim

WORKDIR /app/

RUN groupadd user && useradd --home-dir /app -g user user && chown -R user:user /app
COPY --from=builder --chown=user:user /app/.venv /app/.venv
