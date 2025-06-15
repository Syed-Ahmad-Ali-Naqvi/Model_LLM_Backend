# ollama/Dockerfile
FROM ollama/ollama:latest

# Pull your chosen model at build time
RUN ollama pull llama3.1

# Start the Ollama HTTP server on container launch
ENTRYPOINT ["ollama", "serve", "--host", "0.0.0.0", "--port", "11434"]
