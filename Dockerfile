# Multi-stage build for CPU baseline
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY server /app/server
COPY client /app/client

EXPOSE 5000

ENV HOST=0.0.0.0 PORT=5000 ENABLE_METRICS=1 LOG_LEVEL=INFO

CMD ["uvicorn", "server.main:APP", "--host", "0.0.0.0", "--port", "5000"]


