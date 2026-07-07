# Home project Dockerfile — Windows Docker Desktop compatible.
# Audio capture uses sounddevice (PortAudio/libportaudio2) — no PulseAudio needed.
# GPU build: docker compose build --build-arg GPU=1

ARG GPU=0
FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Removed: pulseaudio — replaced by sounddevice (PortAudio). No daemon needed.
# Removed: pulseaudio-utils, git — not used at runtime.

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install torch + torchaudio together in one layer so pip resolves the full
# dependency graph once. torchaudio is a hard transitive dep of whisperx;
# pinning it here at the same index URL prevents pip from re-downloading torch
# from PyPI during the requirements.txt install step.
# GPU build: set --build-arg GPU=1
ARG GPU
RUN if [ "$GPU" = "1" ]; then \
      pip install --no-cache-dir --timeout 120 --retries 5 \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu121; \
    else \
      pip install --no-cache-dir --timeout 120 --retries 5 \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install everything else. --extra-index-url matches the torch index used above
# so whisperx's solver resolves torch constraints from the same wheel source.
ARG GPU
RUN if [ "$GPU" = "1" ]; then \
      pip install --no-cache-dir --timeout 120 --retries 5 \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121; \
    else \
      pip install --no-cache-dir --timeout 120 --retries 5 \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── App files ─────────────────────────────────────────────────────────────────
COPY app/ ./app/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Runtime dirs ──────────────────────────────────────────────────────────────
RUN mkdir -p /app/output

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
