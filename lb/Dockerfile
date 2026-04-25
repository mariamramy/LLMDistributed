# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-lb.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-lb.txt

COPY proto ./proto
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=. \
    --grpc_python_out=. \
    proto/worker.proto

COPY common ./common
COPY lb ./lb

EXPOSE 8000

CMD ["uvicorn", "lb.main:app", "--host", "0.0.0.0", "--port", "8000"]
