"""
PII Anonymization API Service

FastAPI wrapper for the Presidio + SpaCy German PII detection pipeline.
Designed for containerized deployment.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8080
    
Test:
    curl -X POST http://localhost:8080/anonymize \
        -H "Content-Type: application/json" \
        -d '{"text": "Ich heiße Peter Müller aus Berlin."}'
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scrubb_guard.anonymization_pipeline import PiiPipeline, load_deny_list

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pii-api")

# Global pipeline instance (loaded at startup)
pipeline: Optional[PiiPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the PII pipeline once at startup."""
    global pipeline
    logger.info("Loading PII pipeline (this may take a few seconds)...")
    pipeline = PiiPipeline()
    logger.info("PII pipeline ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="PII Anonymization API",
    description="German PII detection and anonymization using Presidio + SpaCy",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class AnonymizeRequest(BaseModel):
    """Request body for anonymization."""
    text: str = Field(..., description="German text to anonymize", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Ich heiße Peter Müller und wohne in 12345 Berlin."}
            ]
        }
    }


class EntityInfo(BaseModel):
    """Information about a detected entity."""
    entity_type: str = Field(..., description="Type of entity detected")
    text: str = Field(..., description="Original text that was detected")
    score: float = Field(..., description="Confidence score (0-1)")


class AnonymizeResponse(BaseModel):
    """Response from anonymization."""
    anonymized_text: str = Field(..., description="Text with PII replaced by placeholders")
    original_length: int = Field(..., description="Length of original text")
    items_changed: int = Field(..., description="Number of PII items detected and replaced")
    entities: List[EntityInfo] = Field(default_factory=list, description="Details of detected entities")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline_ready: bool
    deny_list_count: int


class ReloadDenyListRequest(BaseModel):
    """Request to reload deny list."""
    entries: Optional[List[str]] = Field(
        None, 
        description="New deny list entries. If null, reloads from config file."
    )


class ReloadDenyListResponse(BaseModel):
    """Response after reloading deny list."""
    success: bool
    entry_count: int
    entries: List[str]


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if the service is healthy and the pipeline is ready."""
    return HealthResponse(
        status="healthy" if pipeline else "initializing",
        pipeline_ready=pipeline is not None,
        deny_list_count=len(pipeline.deny_list) if pipeline else 0
    )


@app.post("/anonymize", response_model=AnonymizeResponse, tags=["Anonymization"])
def anonymize(request: AnonymizeRequest):
    """
    Anonymize German text by detecting and replacing PII.
    
    Detected entity types:
    - **<PERSON>**: Names of people
    - **<ORT>**: Locations, cities, addresses
    - **<ORG>**: Organizations, companies
    - **<PLZ>**: German postal codes (5 digits)
    - **<TEL>**: Phone numbers
    - **<EMAIL>**: Email addresses
    - **<INTERN>**: Custom deny list terms
    - **<DATUM>**: Dates and times
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized")
    
    result = pipeline.process(request.text)
    
    return AnonymizeResponse(
        anonymized_text=result["anonymized_text"],
        original_length=result["original_length"],
        items_changed=result["items_changed"],
        entities=[
            EntityInfo(
                entity_type=e["entity_type"],
                text=e["text"],
                score=e["score"]
            )
            for e in result.get("entities", [])
        ]
    )


@app.post("/deny-list/reload", response_model=ReloadDenyListResponse, tags=["Configuration"])
def reload_deny_list(request: ReloadDenyListRequest = None):
    """
    Reload the deny list from config file or update with new entries.
    
    Use this to hot-reload custom blocked terms without restarting the service.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized")
    
    entries = request.entries if request else None
    updated_list = pipeline.reload_deny_list(entries)
    
    return ReloadDenyListResponse(
        success=True,
        entry_count=len(updated_list),
        entries=updated_list
    )


@app.get("/deny-list", response_model=ReloadDenyListResponse, tags=["Configuration"])
def get_deny_list():
    """Get the current deny list entries."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized")
    
    return ReloadDenyListResponse(
        success=True,
        entry_count=len(pipeline.deny_list),
        entries=pipeline.deny_list
    )


# --- Run directly ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

