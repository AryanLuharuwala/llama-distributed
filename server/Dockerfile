FROM golang:1.25-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY *.go ui.html ./
COPY assets ./assets
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o distpool-server .

# ── runtime ──────────────────────────────────────────────────────────────────
# The custom llama-split-gguf tool lives in a fork of llama.cpp that isn't
# vendored in this repo (it requires API changes to llama_model_params).
# Until that fork is vendored, the splitter binary is provided at runtime via
# an Azure Files upload to /data/bin/ — set DIST_SPLITTER to point at it.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates sqlite3 libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/distpool-server .

# Data volume for SQLite DB, model shards, and release cache
VOLUME ["/data"]

ENV DIST_ADDR=:8080
ENV DIST_DB=/data/distpool.sqlite
ENV DIST_MODELS_DIR=/data/models
ENV DIST_RELEASES_DIR=/data/releases-cache
ENV DIST_SPLITTER=/data/bin/llama-split-gguf
ENV LD_LIBRARY_PATH=/data/bin

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD wget -qO- http://localhost:8080/healthz || exit 1

ENTRYPOINT ["./distpool-server"]
