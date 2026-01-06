# PII Anonymization API Service
# Multi-stage build for smaller final image
#
# Build from project root:
#   docker build -f deploy/anonymization-api/Dockerfile -t pii-anonymizer .

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY deploy/anonymization-api/requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY scrubb_guard/ ./scrubb_guard/
COPY config/ ./config/
COPY deploy/anonymization-api/api.py .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run API server
# - workers=1: Single worker since model is memory-heavy
# - timeout=120: Allow time for cold start model loading
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
