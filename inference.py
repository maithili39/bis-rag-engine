"""Entry point for inference - judges will run this script.

Command: python inference.py --input hidden_private_dataset.json --output team_results.json
"""

import json
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import CHROMA_PERSIST_DIR, DATASET_PDF


def run_inference(input_path: str, output_path: str) -> None:
    """
    Run inference on input queries and save results.
    
    Input JSON format (from judges):
    [
        {
            "id": "PUB-01",
            "query": "product description...",
            "expected_standards": ["IS 269: 1989"]   // may or may not be present
        }
    ]
    
    Output JSON format (strict compliance with eval_script.py):
    [
        {
            "id": "PUB-01",
            "query": "product description...",
            "expected_standards": ["IS 269: 1989"],
            "retrieved_standards": ["IS 269: 1989", "IS 8112: 1989", ...],
            "latency_seconds": 1.24
        }
    ]
    """
    
    print("=" * 60)
    print("BIS Standards Recommendation Engine - Inference")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        print("\n[1/4] Initializing RAG pipeline...")
        pipeline = BISRAGPipeline(persist_dir=CHROMA_PERSIST_DIR)
        
        # Try to load existing vectorstore, otherwise build from PDF
        print("[2/4] Loading vectorstore...")
        if not pipeline.load_existing_vectorstore():
            print("  Vectorstore not found. Building from PDF dataset...")
            if os.path.exists(DATASET_PDF):
                pipeline.initialize_from_pdf(DATASET_PDF)
            else:
                print(f"ERROR: Dataset PDF not found at {DATASET_PDF}")
                print("  Please run: python build_vectorstore.py")
                sys.exit(1)
        else:
            print("  Vectorstore loaded successfully!")
        
        # Load input queries
        print(f"[3/4] Reading queries from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        if not isinstance(input_data, list):
            print("ERROR: Input must be a JSON array")
            sys.exit(1)
        
        num_queries = len(input_data)
        print(f"  Found {num_queries} queries to process")
        
        # Process each query
        print(f"[4/4] Processing queries...")
        results = []
        total_time = 0
        
        for idx, item in enumerate(input_data, 1):
            query_id = item.get("id", f"query_{idx}")
            query = item.get("query", "")
            expected = item.get("expected_standards", [])
            
            if not query:
                print(f"  WARNING: Skipping {query_id}: empty query")
                continue
            
            # Show progress
            print(f"  [{idx}/{num_queries}] {query_id}: {query[:50]}...")
            
            # Process through pipeline
            retrieved_codes, rationale, latency = pipeline.process_query(query, top_k=5)
            total_time += latency
            
            # Build result entry matching eval_script.py expected format
            result = {
                "id": query_id,
                "query": query,
                "retrieved_standards": retrieved_codes,
                "rationale": rationale,
                "latency_seconds": round(latency, 2)
            }
            # Only include expected_standards if present in the input (for eval_script scoring)
            if expected:
                result["expected_standards"] = expected
            results.append(result)
        
        # Save results
        print(f"\nSaving {len(results)} results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Queries processed : {len(results)}")
        print(f"Total time        : {total_time:.2f}s")
        print(f"Avg latency       : {total_time/max(len(results),1):.2f}s/query")
        print(f"Output file       : {output_path}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standards Recommendation Engine Inference"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file with queries"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file for results"
    )
    args = parser.parse_args()
    run_inference(args.input, args.output)


if __name__ == "__main__":
    main()
