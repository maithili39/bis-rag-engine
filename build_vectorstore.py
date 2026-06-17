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
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import CHROMA_PERSIST_DIR, DATASET_PDF, DATA_DIR


def main():
    print("=" * 60)
    print("BIS RAG Engine - Vectorstore Builder")
    print("=" * 60)
    
    # Scan for PDF files in the data directory (Phase 6)
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"\nERROR: No PDF files found in data directory: {DATA_DIR}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF(s) to process:")
    for pdf in pdf_files:
        print(f"  - {os.path.basename(pdf)}")
    print(f"Output:  {CHROMA_PERSIST_DIR}")
    
    # Remove old vectorstore if exists
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"\nRemoving old vectorstore at {CHROMA_PERSIST_DIR}...")
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    # Build
    start = time.time()
    pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
    pipeline.initialize_from_pdfs(pdf_files)
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

    # ── Diff known_codes.json against any previously committed version ───────
    codes_path = os.path.join(os.path.dirname(CHROMA_PERSIST_DIR), "known_codes.json")
    old_codes_path = codes_path + ".old"
    if os.path.exists(old_codes_path):
        with open(old_codes_path, encoding="utf-8") as f:
            old_codes = set(json.load(f))
        with open(codes_path, encoding="utf-8") as f:
            new_codes = set(json.load(f))

        added = sorted(new_codes - old_codes)
        removed = sorted(old_codes - new_codes)

        print(f"\n--- known_codes.json diff (vs .old backup) ---")
        if added:
            print(f"  + {len(added)} added:")
            for c in added[:20]:
                print(f"      + {c}")
            if len(added) > 20:
                print(f"      ... and {len(added) - 20} more")
        if removed:
            print(f"  - {len(removed)} removed:")
            for c in removed[:20]:
                print(f"      - {c}")
            if len(removed) > 20:
                print(f"      ... and {len(removed) - 20} more")
        if not added and not removed:
            print("  No changes (artifacts were already clean).")
        print("----------------------------------------------")
    else:
        # Back up the old file before first clean build so the diff works next time
        import shutil as _shutil
        if os.path.exists(codes_path):
            _shutil.copy2(codes_path, old_codes_path)
            print(f"\n(Backed up previous known_codes.json to {old_codes_path})")
    
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
