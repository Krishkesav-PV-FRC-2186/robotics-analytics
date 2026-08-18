FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# API/dashboard import paths do not load YOLO at startup. Skip heavy CV wheels
# so compose images stay bootable; install the full requirements.txt locally for vision.
RUN grep -vE '^(ultralytics|supervision|opencv-python|easyocr)$' requirements.txt \
        > /tmp/runtime-requirements.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/runtime-requirements.txt

COPY src ./src
COPY scripts ./scripts

EXPOSE 8000 8501
