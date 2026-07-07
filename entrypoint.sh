#!/bin/sh
# Audio Notes — container entrypoint
# PulseAudio is not used. Audio capture runs via sounddevice (PortAudio).
# Nothing to start before uvicorn — just exec it directly.

set -e

echo "[entrypoint] Starting Audio Notes..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
