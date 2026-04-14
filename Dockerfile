# ────────────────────────────────────────────────────────────
# CEG4195 Lab 2 – Dockerfile
# House Segmentation API
# ────────────────────────────────────────────────────────────
FROM python:3.9-slim

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py     .
COPY scripts/   scripts/

# Model directory (weights mounted at runtime via volume)
RUN mkdir -p models

# Non-root user for security best practice
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Gunicorn for production  (2 workers, 120 s timeout for inference)
CMD ["gunicorn", \
     "--bind",    "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "app:app"]
