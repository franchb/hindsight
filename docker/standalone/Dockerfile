# syntax=docker/dockerfile:1
# Hindsight Docker Image — Chainguard-based slim builds
#
# Targets:
#   api-only   — Distroless Python runtime (no shell, ~400MB)
#   standalone — wolfi-base with Python + Node.js + bash (~500MB)
#   cp-only    — Chainguard Node.js distroless, Control Plane only
#
# Build:
#   podman build -t hindsight-api --target api-only -f docker/standalone/Dockerfile .
#   podman build -t hindsight --target standalone -f docker/standalone/Dockerfile .
#
# Note: Free-tier Chainguard images track latest Python (currently 3.14).
# If upstream bumps to 3.15, builds may need adjustment.

# =============================================================================
# Stage: API Builder (Chainguard dev — has uv, pip, apk, bash)
# =============================================================================
FROM cgr.dev/chainguard/python:latest-dev AS api-builder

USER root
WORKDIR /app/api

# Install build dependencies for native C extensions (if any wheel lacks a prebuilt)
RUN apk add --no-cache build-base

# Copy dependency files first for Docker layer caching
COPY hindsight-api/pyproject.toml hindsight-api/README.md ./

# Install dependencies only (project itself installed after source copy)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

# Copy source code (includes alembic migrations inside hindsight_api/)
COPY hindsight-api/hindsight_api ./hindsight_api

# Install the project package (editable; deps already present)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

# Pre-download tiktoken encoding (required at runtime for token counting)
ENV TIKTOKEN_CACHE_DIR=/tiktoken_cache
RUN /app/api/.venv/bin/python -c "\
import tiktoken; \
tiktoken.get_encoding('cl100k_base'); \
print('Tiktoken encoding cached')"

# Clean venv: remove tests, caches, and build artifacts
RUN find .venv -type d \( -name tests -o -name test -o -name __pycache__ \) -exec rm -rf {} + 2>/dev/null; \
    find .venv \( -name '*.pyc' -o -name '*.pyo' -o -name '*.c' -o -name '*.h' \) -delete 2>/dev/null; \
    find .venv -name 'LICENSE*' -delete 2>/dev/null; \
    true

# =============================================================================
# Stage: SDK Builder (needed for Control Plane)
# =============================================================================
FROM node:20-slim AS sdk-builder

WORKDIR /app

# Copy root package files for npm workspaces
COPY package.json package-lock.json ./
COPY hindsight-clients/typescript/ ./hindsight-clients/typescript/

# Install and build SDK using workspace (--ignore-scripts skips git hooks setup)
RUN npm ci --ignore-scripts -w @vectorize-io/hindsight-client
RUN npm run build -w @vectorize-io/hindsight-client

# =============================================================================
# Stage: Control Plane Builder
# =============================================================================
FROM node:20-slim AS cp-builder

# Create directory structure matching the monorepo layout
# Required because build:standalone script expects .next/standalone/memory-poc/hindsight-control-plane
WORKDIR /app/memory-poc/hindsight-control-plane

# Install Control Plane dependencies
# Only copy package.json (not package-lock.json) to ensure npm installs
# correct platform-specific native bindings for lightningcss/tailwindcss
COPY hindsight-control-plane/package.json ./
# Remove the file: dependency on SDK (we'll copy it directly later)
RUN sed -i '/"@vectorize-io\/hindsight-client":/d' package.json
RUN npm install

# Copy Control Plane source (excluding node_modules via .dockerignore)
COPY hindsight-control-plane/ ./
# Remove package-lock.json to avoid conflicts with installed native bindings
# Also remove the file: dependency from package.json (restored by COPY above)
RUN rm -f package-lock.json && sed -i '/"@vectorize-io\/hindsight-client":/d' package.json

# Copy built SDK directly into node_modules (more reliable than npm link in Docker)
COPY --from=sdk-builder /app/hindsight-clients/typescript ./node_modules/@vectorize-io/hindsight-client

# Accept base path as build argument for reverse proxy deployments
ARG NEXT_PUBLIC_BASE_PATH=""

# Build Control Plane
RUN npm exec -- next build

# Create standalone directory structure manually
# Note: Must exclude node_modules from find to avoid wrong server.js from next/dist/experimental/testmode/
# Note: Must explicitly copy .next since glob * doesn't match hidden directories
RUN STANDALONE_ROOT=$(find .next/standalone -path '*/node_modules' -prune -o -name 'server.js' -print | head -1 | xargs dirname) && \
    mkdir -p standalone && \
    cp -r "$STANDALONE_ROOT"/* standalone/ && \
    cp -r "$STANDALONE_ROOT"/.next standalone/.next && \
    if [ -d ".next/standalone/node_modules" ] && [ "$STANDALONE_ROOT" != ".next/standalone" ]; then \
      cp -r .next/standalone/node_modules standalone/node_modules; \
    fi && \
    cp -r .next/static standalone/.next/static && \
    mkdir -p standalone/public && \
    cp -r public/* standalone/public/ 2>/dev/null || true && \
    test -f standalone/server.js || (echo "ERROR: server.js missing!" && exit 1) && \
    test -f standalone/.next/BUILD_ID || (echo "ERROR: BUILD_ID missing!" && exit 1)

# =============================================================================
# Stage: api-only (Chainguard Python distroless — no shell, nonroot UID 65532)
# =============================================================================
FROM cgr.dev/chainguard/python:latest AS api-only

WORKDIR /app

# Copy API with virtual environment from builder
COPY --from=api-builder /app/api /app/api

# Copy tiktoken cache (root-owned but world-readable; tiktoken only reads)
COPY --from=api-builder /tiktoken_cache /home/nonroot/.tiktoken

# Copy Python entrypoint (no shell available in distroless)
COPY docker/standalone/entrypoint.py /app/entrypoint.py

ENV PATH="/app/api/.venv/bin:${PATH}" \
    TIKTOKEN_CACHE_DIR=/home/nonroot/.tiktoken \
    HINDSIGHT_API_HOST=0.0.0.0 \
    HINDSIGHT_API_PORT=8888 \
    HINDSIGHT_API_LOG_LEVEL=info \
    HINDSIGHT_ENABLE_API=true \
    HINDSIGHT_ENABLE_CP=false \
    PYTHONUNBUFFERED=1

EXPOSE 8888

ENTRYPOINT ["python", "/app/entrypoint.py"]

# =============================================================================
# Stage: cp-only (Chainguard Node.js distroless — Control Plane only)
# =============================================================================
FROM cgr.dev/chainguard/node:latest AS cp-only

WORKDIR /app/control-plane

# Copy Control Plane standalone build (includes .next/static and public/)
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/standalone ./

EXPOSE 9999

ENV NODE_ENV=production \
    HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888 \
    PORT=9999

CMD ["server.js"]

# =============================================================================
# Stage: standalone (Chainguard wolfi-base — Python + Node.js + bash)
# =============================================================================
FROM cgr.dev/chainguard/wolfi-base AS standalone

# Install runtime dependencies (wolfi-base provides apk + ash; add bash explicitly)
RUN apk add --no-cache python-3.14 nodejs-20 curl bash

WORKDIR /app

# Copy API with virtual environment from builder
COPY --from=api-builder /app/api /app/api

# Copy tiktoken cache
COPY --from=api-builder /tiktoken_cache /home/nonroot/.tiktoken

# Copy built SDK
COPY --from=sdk-builder /app/hindsight-clients/typescript /app/sdk

# Copy Control Plane standalone build
WORKDIR /app/control-plane
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/standalone ./
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/.next/static ./.next/static
COPY --from=cp-builder /app/memory-poc/hindsight-control-plane/public ./public

WORKDIR /app

# Copy startup script
COPY docker/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

# Run as nonroot (UID 65532, built into wolfi-base)
USER nonroot

ENV PATH="/app/api/.venv/bin:${PATH}" \
    TIKTOKEN_CACHE_DIR=/home/nonroot/.tiktoken \
    HINDSIGHT_API_HOST=0.0.0.0 \
    HINDSIGHT_API_PORT=8888 \
    HINDSIGHT_API_LOG_LEVEL=info \
    NODE_ENV=production \
    HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888 \
    HINDSIGHT_ENABLE_API=true \
    HINDSIGHT_ENABLE_CP=true \
    PYTHONUNBUFFERED=1

EXPOSE 8888 9999

CMD ["/app/start-all.sh"]

# Default target
FROM standalone
