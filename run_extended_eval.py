"""Extended evaluation runner — A/B compares hybrid vs cross-encoder reranking.

Usage:
    # Standard hybrid-only run (no extra deps needed):
    python run_extended_eval.py

    # With cross-encoder reranking (requires first run to download model ~90 MB):
    USE_CROSS_ENCODER=1 python run_extended_eval.py

    # Run both modes back to back for an A/B comparison table:
    python run_extended_eval.py --ab

Output files:
    data/extended_test_results.json          — hybrid results
    data/extended_test_results_ce.json       — cross-encoder results (--ab mode)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import BISRAGPipeline
from config import DATASET_PDF


# ── Metric helpers ────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return str(s).replace(" ", "").lower()


def compute_metrics(results: list) -> dict:
    """Compute Hit@3, MRR@5, and latency stats over a results list.

    Queries with an empty expected_standards list are treated as out-of-scope
    negatives: they contribute to latency stats but NOT to Hit@3/MRR
    (returning nothing for an out-of-scope query is correct behaviour, not a miss).
    """
    in_scope = [r for r in results if r.get("expected_standards")]
    out_of_scope = [r for r in results if not r.get("expected_standards")]

    hits3 = 0
    mrr_sum = 0.0

    for item in in_scope:
        expected = {_norm(s) for s in item["expected_standards"]}
        retrieved = [_norm(s) for s in item.get("retrieved_standards", [])]

        if any(s in expected for s in retrieved[:3]):
            hits3 += 1

        for rank, s in enumerate(retrieved[:5], 1):
            if s in expected:
                mrr_sum += 1.0 / rank
                break

    n = len(in_scope) or 1
    all_latencies = [r.get("latency_seconds", 0) for r in results]

    return {
        "total_queries": len(results),
        "in_scope_queries": len(in_scope),
        "out_of_scope_queries": len(out_of_scope),
        "hit_rate_at_3": round(hits3 / n * 100, 2),
        "mrr_at_5": round(mrr_sum / n, 4),
        "avg_latency_s": round(sum(all_latencies) / max(len(all_latencies), 1), 3),
        "max_latency_s": round(max(all_latencies, default=0), 3),
        "hits_at_3": hits3,
    }


def print_metrics(label: str, m: dict) -> None:
    width = 44
    print("=" * width)
    print(f"  {label}")
    print("=" * width)
    print(f"  Total queries      : {m['total_queries']} ({m['in_scope_queries']} in-scope, {m['out_of_scope_queries']} negative)")
    print(f"  Hit Rate @3        : {m['hit_rate_at_3']:.2f}%  (target: >80%)")
    print(f"  MRR @5             : {m['mrr_at_5']:.4f}   (target: >0.70)")
    print(f"  Avg latency        : {m['avg_latency_s']:.2f}s   (target: <5s)")
    print(f"  Max latency        : {m['max_latency_s']:.2f}s")
    print("=" * width)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_eval(pipeline: BISRAGPipeline, test_set: list, label: str, output_path: str) -> dict:
    print(f"\n[{label}] Processing {len(test_set)} queries...")
    results = []
    for i, item in enumerate(test_set, 1):
        codes, _, latency = pipeline.process_query(item["query"], top_k=5)
        result = {
            "id": item["id"],
            "query": item["query"],
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": codes,
            "latency_seconds": round(latency, 3),
        }
        results.append(result)
        if i % 10 == 0:
            print(f"  {i}/{len(test_set)} done...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {output_path}")
    return compute_metrics(results)


def main():
    parser = argparse.ArgumentParser(description="Extended A/B evaluation runner")
    parser.add_argument(
        "--ab",
        action="store_true",
        help="Run both hybrid-only and cross-encoder modes and print a comparison table.",
    )
    parser.add_argument(
        "--input",
        default="data/extended_test_set.json",
        help="Path to the test set JSON file.",
    )
    args = parser.parse_args()

    print("=" * 44)
    print("  BIS RAG Extended Evaluation Runner")
    print("=" * 44)

    # Load test set
    with open(args.input, encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"\nTest set: {args.input} ({len(test_set)} queries)")

    # ── Hybrid-only run ──────────────────────────────────────────────────────
    # Force cross-encoder off for the baseline run
    os.environ.pop("USE_CROSS_ENCODER", None)
    # Re-import config to pick up the cleared env var (module may already be loaded)
    import importlib, config as _cfg
    _cfg.USE_CROSS_ENCODER = False

    pipeline = BISRAGPipeline()
    if not pipeline.load_existing_vectorstore():
        print("Vectorstore not found — building from PDF (this takes a few minutes)...")
        pipeline.initialize_from_pdf(DATASET_PDF)

    hybrid_metrics = run_eval(
        pipeline, test_set,
        label="Hybrid (BM25 + Semantic)",
        output_path="data/extended_test_results.json",
    )

    if not args.ab:
        print_metrics("HYBRID RESULTS", hybrid_metrics)
        return

    # ── Cross-encoder run ────────────────────────────────────────────────────
    os.environ["USE_CROSS_ENCODER"] = "1"
    _cfg.USE_CROSS_ENCODER = True

    # Re-create retriever so it picks up USE_CROSS_ENCODER=True
    pipeline2 = BISRAGPipeline()
    pipeline2.load_existing_vectorstore()

    ce_metrics = run_eval(
        pipeline2, test_set,
        label="Cross-encoder (BM25 + Semantic + CE)",
        output_path="data/extended_test_results_ce.json",
    )

    # ── A/B comparison table ─────────────────────────────────────────────────
    print("\n")
    print_metrics("BASELINE — Hybrid only", hybrid_metrics)
    print()
    print_metrics("RERANKED — Cross-encoder", ce_metrics)

    print("\nA/B delta:")
    hr_delta = ce_metrics["hit_rate_at_3"] - hybrid_metrics["hit_rate_at_3"]
    mrr_delta = ce_metrics["mrr_at_5"] - hybrid_metrics["mrr_at_5"]
    lat_delta = ce_metrics["avg_latency_s"] - hybrid_metrics["avg_latency_s"]
    print(f"  Hit Rate @3 : {hr_delta:+.2f}%")
    print(f"  MRR @5      : {mrr_delta:+.4f}")
    print(f"  Avg latency : {lat_delta:+.2f}s")


if __name__ == "__main__":
    main()
