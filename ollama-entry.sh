#!/usr/bin/env sh
set -e

# Pull the model into the shared volume (if missing)
ollama pull llama3.1

# Launch the Ollama HTTP server
ollama serve --host 0.0.0.0 --port 11434
