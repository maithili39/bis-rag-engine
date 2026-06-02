"""Main RAG pipeline orchestrating retrieval and recommendation generation."""

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import BISDataProcessor
from retriever import BISRetriever
from llm_generator import RecommendationGenerator
from config import TOP_K_RESULTS, TOP_K_RETRIEVAL, CHROMA_PERSIST_DIR, DATASET_PDF


class BISRAGPipeline:
    """End-to-end RAG pipeline for BIS standard recommendations.

    Pipeline: Query -> Embed -> Retrieve -> Extract Codes -> Filter -> Return
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.data_processor = BISDataProcessor()
        self.retriever = BISRetriever(persist_dir=persist_dir)
        self.generator = RecommendationGenerator()
        self.known_codes: Set[str] = set()
        self.query_history = []

    def initialize_from_pdf(self, pdf_path: str = DATASET_PDF) -> None:
        """Initialize the pipeline by processing the BIS SP 21 PDF."""
        print(f"\nProcessing PDF: {pdf_path}")
        documents, known_codes = self.data_processor.process_pdf(pdf_path)

        self.known_codes = known_codes
        self.generator.set_known_codes(known_codes)

        # Save known codes for later use
        codes_path = os.path.join(os.path.dirname(self.persist_dir), "known_codes.json")
        with open(codes_path, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(known_codes)), f, indent=2)
        print(f"  Saved {len(known_codes)} known codes to {codes_path}")

        print("\nBuilding vectorstore...")
        self.retriever.build_vectorstore(documents)
        print("Pipeline initialized successfully!")

    def load_existing_vectorstore(self) -> bool:
        """Load an existing vectorstore and known codes."""
        loaded = self.retriever.load_vectorstore()
        if loaded:
            # Try to load known codes
            codes_path = os.path.join(os.path.dirname(self.persist_dir), "known_codes.json")
            if os.path.exists(codes_path):
                with open(codes_path, 'r', encoding='utf-8') as f:
                    self.known_codes = set(json.load(f))
                self.generator.set_known_codes(self.known_codes)
                print(f"  Loaded {len(self.known_codes)} known standard codes")
        return loaded

    def process_query(self, query: str, top_k: int = TOP_K_RESULTS) -> Tuple[List[str], str, float]:
        """
        Process a single query through the full RAG pipeline.

        Returns:
            Tuple of (list of standard code strings, rationale string, total latency)
        """
        pipeline_start = time.time()

        # Step 1: Retrieve relevant documents
        retrieved_docs, retrieval_latency = self.retriever.retrieve(query, k=TOP_K_RETRIEVAL)

        # Step 2: Extract and rank standard codes
        codes, gen_latency = self.generator.generate_recommendations(
            query, retrieved_docs, top_k=top_k
        )

        # Step 3: Generate rationale for the recommendations
        rationale = self.generator.generate_rationale(query, codes)

        total_latency = time.time() - pipeline_start

        # Track history
        self.query_history.append({
            "query": query,
            "results": codes,
            "latency": total_latency
        })

        return codes, rationale, total_latency

    def process_query_detailed(self, query: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """Process query and return detailed results (for web UI)."""
        pipeline_start = time.time()

        # Retrieve with scores
        results_with_scores, retrieval_latency = self.retriever.retrieve_with_scores(query, k=TOP_K_RETRIEVAL)

        # Extract codes
        docs = [doc for doc, _ in results_with_scores]
        [float(score) for _, score in results_with_scores]

        codes, _ = self.generator.generate_recommendations(query, docs, top_k=top_k)

        total_latency = time.time() - pipeline_start

        # Build detailed results
        details = []
        seen = set()
        for doc, score in results_with_scores:
            code = doc.metadata.get("standard_code", "")
            if code and code not in seen:
                seen.add(code)
                details.append({
                    "code": code,
                    "title": doc.metadata.get("title", ""),
                    "score": round(1 - score, 4),  # Convert distance to similarity
                    "snippet": doc.page_content[:300]
                })

        return {
            "query": query,
            "recommended_standards": codes[:top_k],
            "details": details[:top_k],
            "latency_seconds": round(total_latency, 3),
            "num_retrieved": len(results_with_scores)
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "vectorstore": self.retriever.get_vectorstore_stats(),
            "known_standards": len(self.known_codes),
            "queries_processed": len(self.query_history),
            "avg_retrieval_latency": self.retriever.get_average_latency(),
        }
        if self.query_history:
            latencies = [q["latency"] for q in self.query_history]
            stats["avg_pipeline_latency"] = sum(latencies) / len(latencies)
        return stats

    def reset_history(self) -> None:
        """Reset query history."""
        self.query_history = []
        self.retriever.reset_latency_tracking()
