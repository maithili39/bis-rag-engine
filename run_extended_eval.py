"""Run the pipeline over the extended test set, emit results, and evaluate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import DATASET_PDF

pipeline = BISRAGPipeline()
if not pipeline.load_existing_vectorstore():
    pipeline.initialize_from_pdf(DATASET_PDF)

with open("data/extended_test_set.json", encoding="utf-8") as f:
    test_set = json.load(f)

results = []
for item in test_set:
    codes, _, latency = pipeline.process_query(item["query"], top_k=5)
    results.append({
        "id": item["id"],
        "query": item["query"],
        "expected_standards": item["expected_standards"],
        "retrieved_standards": codes,
        "latency_seconds": round(latency, 3),
    })

with open("data/extended_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} results to data/extended_test_results.json")
