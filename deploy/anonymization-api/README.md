# PII Anonymization API - Deployment

Containerized deployment of the German PII anonymization pipeline.

## Quick Start

From the **project root** directory:

```bash
# Using Docker Compose (recommended)
docker compose -f deploy/anonymization-api/docker-compose.yml up --build

# Or using Docker directly
docker build -f deploy/anonymization-api/Dockerfile -t pii-anonymizer .
docker run -p 8080:8080 pii-anonymizer
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + pipeline status |
| `POST` | `/anonymize` | Anonymize German text |
| `GET` | `/deny-list` | View current deny list |
| `POST` | `/deny-list/reload` | Hot-reload deny list |
| `GET` | `/docs` | Interactive Swagger documentation |

## Test the API

```bash
# Health check
curl http://localhost:8080/health

# Anonymize text
curl -X POST http://localhost:8080/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Ich heiße Peter Müller und wohne in 12345 Berlin."}'

# Expected response:
# {
#   "anonymized_text": "Ich heiße <PERSON> und wohne in <PLZ> <ORT>.",
#   "original_length": 51,
#   "items_changed": 3,
#   "entities": [...]
# }
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage container build |
| `docker-compose.yml` | Local development setup |
| `requirements-api.txt` | Python dependencies (minimal) |
| `api.py` | FastAPI service wrapper |
| `.dockerignore` | Excludes unnecessary files |

## Resource Requirements

- **Memory**: 1-1.5 GB (SpaCy model is ~500MB)
- **CPU**: 1-2 vCPUs
- **Image size**: ~800 MB
- **Cold start**: 5-15 seconds (model loading)

## VM Deployment

```bash
# Build and push to registry
docker build -f deploy/anonymization-api/Dockerfile -t your-registry/pii-anonymizer:latest .
docker push your-registry/pii-anonymizer:latest

# On VM
docker pull your-registry/pii-anonymizer:latest
docker run -d --name pii-anonymizer -p 8080:8080 --restart unless-stopped \
  your-registry/pii-anonymizer:latest
```

