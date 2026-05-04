"""Configuration management for BIS RAG Engine."""

import os
from dotenv import load_dotenv

load_dotenv()

# Disable ChromaDB anonymous telemetry (suppresses posthog capture() warnings)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Embedding Configuration (FREE - no API key needed)
EMBEDDING_MODEL = "all-mpnet-base-v2"  # HuggingFace sentence-transformers (768-dim, optimized for information retrieval)

# Optional: OpenAI for LLM-enhanced reranking (ONLY used in web UI, NOT in hackathon evaluation)
# The system works perfectly fine without this - it falls back to intelligent rule-based reranking
# To enable: Set OPENAI_API_KEY environment variable with your OpenAI API key
# Leave empty or unset for fully free operation (recommended for hackathon evaluation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-3.5-turbo"

# RAG Configuration
CHUNK_SIZE = 1000       # Optimized chunk size for technical documents
CHUNK_OVERLAP = 250     # Increased overlap for better context preservation at boundaries
TOP_K_RETRIEVAL = 12    # Reduced from 20 — enough candidates for top-5 output with less overhead
TOP_K_RESULTS = 5       # Return top 5 standards

# Database Configuration
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chromadb")
COLLECTION_NAME = "bis_standards"

# PDF Dataset
DATASET_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.pdf")

# Building Materials Categories
BUILDING_MATERIALS_CATEGORIES = [
    "cement", "steel", "concrete", "aggregates", "bricks", "tiles",
    "glass", "paint", "insulation", "roofing", "timber", "lime",
    "mortar", "pipes", "blocks", "sheets", "plywood", "doors",
    "windows", "flooring", "waterproofing", "admixtures",
]

# Evaluation Thresholds
HIT_RATE_TARGET = 80
MRR_TARGET = 0.7
LATENCY_THRESHOLD = 5.0
EVAL_HIT_RATE_K = 3
EVAL_MRR_K = 5

# Hybrid Search Weights (Semantic + BM25)
SEMANTIC_WEIGHT = 0.4   # Semantic embeddings for overall standard relevance
BM25_WEIGHT = 0.6       # BM25 keyword matching is critical for technical terminology (codes, standards)
