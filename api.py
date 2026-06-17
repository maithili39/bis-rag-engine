"""FastAPI REST API for the BIS Standards Recommendation Engine.

Usage:
    pip install fastapi uvicorn
    python api.py                    # runs on http://localhost:8000
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query          — recommend standards for a product description
    GET  /health         — liveness check + pipeline stats
    GET  /standards      — list all known standard codes
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag_pipeline import BISRAGPipeline
from src.config import CHROMA_PERSIST_DIR, DATASET_PDF

_pipeline: Optional[BISRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
    if not _pipeline.load_existing_vectorstore():
        if os.path.exists(DATASET_PDF):
            print("Vectorstore not found — building from PDF...")
            _pipeline.initialize_from_pdf(DATASET_PDF)
        else:
            raise RuntimeError(
                f"Neither vectorstore nor dataset PDF found. "
                f"Run: python build_vectorstore.py"
            )
    yield
    _pipeline = None


app = FastAPI(
    title="BIS Standards Recommendation API",
    description="Recommends Bureau of Indian Standards codes for building material product descriptions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS using environment variable, default to allowing all
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",")] if cors_origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add X-Process-Time header for latency visibility
import time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response

# TODO: Add rate limiting for production
# Full implementation requires slowapi (e.g. @limiter.limit("10/minute"))
# Deferred unless specifically requested for the hackathon.


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Product or material description")
    top_k: int = Field(5, ge=1, le=20, description="Number of standards to return")


class QueryResponse(BaseModel):
    query: str
    recommended_standards: List[str]
    rationale: str
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    pipeline_stats: dict


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    codes, rationale, latency = _pipeline.process_query(req.query, top_k=req.top_k)
    return QueryResponse(
        query=req.query,
        recommended_standards=codes,
        rationale=rationale,
        latency_seconds=round(latency, 3),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _pipeline is None:
        return HealthResponse(status="starting", pipeline_stats={})
    return HealthResponse(status="ok", pipeline_stats=_pipeline.get_pipeline_stats())


@app.get("/standards")
def list_standards() -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {"standards": sorted(_pipeline.known_codes), "count": len(_pipeline.known_codes)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
