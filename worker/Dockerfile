# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-worker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-worker.txt

COPY proto ./proto
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=. \
    --grpc_python_out=. \
    proto/worker.proto

COPY common ./common
COPY worker ./worker

CMD ["python", "-m", "worker.server"]
