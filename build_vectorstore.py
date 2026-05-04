"""Build the ChromaDB vectorstore from the BIS SP 21 PDF.

Run this once before inference:
    python build_vectorstore.py

This will:
1. Parse the BIS SP 21 PDF (data/dataset.pdf)
2. Extract individual standard entries
3. Create embeddings using sentence-transformers (free, no API key)
4. Persist the vectorstore to ./chromadb/
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import CHROMA_PERSIST_DIR, DATASET_PDF


def main():
    print("=" * 60)
    print("BIS RAG Engine - Vectorstore Builder")
    print("=" * 60)
    
    # Check for dataset
    if not os.path.exists(DATASET_PDF):
        print(f"\nERROR: Dataset PDF not found at: {DATASET_PDF}")
        print("Please place the BIS SP 21 PDF at data/dataset.pdf")
        sys.exit(1)

    print(f"\nDataset: {DATASET_PDF}")
    print(f"Output:  {CHROMA_PERSIST_DIR}")
    
    # Remove old vectorstore if exists
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"\nRemoving old vectorstore at {CHROMA_PERSIST_DIR}...")
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    # Build
    start = time.time()
    pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
    pipeline.initialize_from_pdf(DATASET_PDF)
    elapsed = time.time() - start
    
    # Print stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n{'=' * 60}")
    print("BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Time elapsed     : {elapsed:.1f}s")
    print(f"Documents stored : {stats['vectorstore'].get('total_documents', 'N/A')}")
    print(f"Known standards  : {stats['known_standards']}")
    print(f"Persist dir      : {CHROMA_PERSIST_DIR}")
    print(f"{'=' * 60}")
    
    # Quick test
    print("\n--- Quick Test ---")
    test_queries = [
        "Ordinary Portland Cement 33 grade",
        "coarse and fine aggregates for concrete",
        "Portland slag cement",
    ]
    for q in test_queries:
        codes, rationale, latency = pipeline.process_query(q)
        print(f"  Q: {q}")
        print(f"  -> {codes[:3]} ({latency:.2f}s)\n")


if __name__ == "__main__":
    main()
