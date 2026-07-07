#!/bin/sh
# ollama-init.sh — runs once at first compose up to pull gemma3:270m.
# Mounted into the model-loader sidecar container.
# Uses the ollama REST API so no shell variable escaping issues exist.
# On subsequent starts the model is already in the volume — pull is instant.

set -e

MODEL="gemma3:270m"
OLLAMA_HOST="http://ollama:11434"

echo "[model-loader] Waiting for ollama to be ready..."
until wget -qO- "${OLLAMA_HOST}/api/version" >/dev/null 2>&1; do
  sleep 2
done
echo "[model-loader] ollama is up. Pulling ${MODEL}..."

wget -qO- \
  --post-data="{\"name\":\"${MODEL}\"}" \
  --header="Content-Type: application/json" \
  "${OLLAMA_HOST}/api/pull" | tail -1

echo "[model-loader] ${MODEL} pull complete."
