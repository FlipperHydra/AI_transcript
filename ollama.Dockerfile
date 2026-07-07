# ollama.Dockerfile — bakes phi3:mini into the image at build time.
# On first build this downloads ~2.3GB once. Every subsequent `docker compose up`
# starts instantly with the model already present — no pull on startup.
FROM ollama/ollama:latest

# Start the server in the background, poll until it responds, pull the model,
# then shut down cleanly. Polling replaces the fragile `sleep 5` — on slow
# machines the server may not be ready in 5 s, causing a silent pull failure.
RUN /bin/sh -c '\
  ollama serve & \
  SERVER_PID=$! && \
  echo "Waiting for ollama to be ready..." && \
  i=0; while [ $i -lt 30 ]; do \
    if wget -qO- http://localhost:11434/ >/dev/null 2>&1; then \
      echo "ollama ready after ${i}s"; break; \
    fi; \
    sleep 1; i=$((i+1)); \
  done && \
  ollama pull phi3:mini && \
  kill $SERVER_PID && \
  wait $SERVER_PID 2>/dev/null; \
  true'
