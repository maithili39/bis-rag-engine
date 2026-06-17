"""Hybrid retrieval for BIS standards: dense embeddings + BM25 keyword search.

The corpus (a few hundred standard entries) is small enough that a full
ChromaDB instance is unnecessary overhead — instead we keep an in-memory
NumPy matrix of L2-normalised embeddings and compute cosine similarity with a
single matrix-vector product. This has zero native dependencies (no compiler
needed), installs and deploys anywhere, and starts up in well under a second.
"""

import hashlib
import os
import pickle
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BM25_WEIGHT, CHROMA_PERSIST_DIR, CROSS_ENCODER_MODEL, EMBEDDING_MODEL, MIN_CONFIDENCE_SCORE, SEMANTIC_WEIGHT, TOP_K_RETRIEVAL, USE_CROSS_ENCODER  # noqa: E402


class BISRetriever:
    """Retrieves relevant BIS standards using hybrid search (dense + BM25).

    Dense vectors come from a free HuggingFace sentence-transformers model
    (no API key). Cosine similarity is computed in NumPy; BM25 supplies
    keyword matching that is critical for technical codes and terminology.
    """

    EMBEDDINGS_FILE = "embeddings.npy"
    STORE_FILE = "store.pkl"

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.doc_embeddings: np.ndarray | None = None   # (N, dim), L2-normalised
        self.all_documents: List[Document] = []
        self.bm25: BM25Okapi | None = None
        self.retrieval_times: List[float] = []
        self._embeddings_path = os.path.join(persist_dir, self.EMBEDDINGS_FILE)
        self._store_path = os.path.join(persist_dir, self.STORE_FILE)
        # Optional cross-encoder: loaded lazily to avoid startup cost when disabled
        self._cross_encoder = None
        if USE_CROSS_ENCODER:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                print(f"  Cross-encoder loaded: {CROSS_ENCODER_MODEL}")
            except Exception as e:
                print(f"  Warning: could not load cross-encoder ({e}). Falling back to hybrid ranking.")

    # ------------------------------------------------------------------ #
    # Build / persist / load                                             #
    # ------------------------------------------------------------------ #

    def build_vectorstore(self, documents: List[Document]) -> None:
        """Embed documents, build the BM25 index, and persist both to disk."""
        if not documents:
            raise ValueError("Cannot build a vectorstore from zero documents.")

        print(f"  Embedding {len(documents)} documents...")
        texts = [doc.page_content for doc in documents]
        self.doc_embeddings = np.asarray(
            self.embeddings.embed_documents(texts), dtype=np.float32
        )
        self.all_documents = documents

        tokenized_docs = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_docs)

        os.makedirs(self.persist_dir, exist_ok=True)
        np.save(self._embeddings_path, self.doc_embeddings)
        with open(self._store_path, "wb") as f:
            pickle.dump(
                {
                    "documents": [(d.page_content, d.metadata) for d in documents],
                    "bm25": self.bm25,
                    "embedding_model": EMBEDDING_MODEL,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"  Vectorstore persisted to {self.persist_dir}")
        print(f"  Indexed {len(documents)} documents ({self.doc_embeddings.shape[1]}-dim)")

    def load_vectorstore(self) -> bool:
        """Load a persisted vectorstore. Returns True on success."""
        if not (os.path.exists(self._embeddings_path) and os.path.exists(self._store_path)):
            return False
        try:
            self.doc_embeddings = np.load(self._embeddings_path)
            with open(self._store_path, "rb") as f:
                store = pickle.load(f)
            self.all_documents = [
                Document(page_content=t, metadata=m) for t, m in store["documents"]
            ]
            self.bm25 = store["bm25"]
            if len(self.all_documents) == 0 or self.doc_embeddings.shape[0] == 0:
                return False
            print(f"  Loaded vectorstore with {len(self.all_documents)} documents")
            return True
        except Exception as e:
            print(f"  Error loading vectorstore: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #

    def _embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self.embeddings.embed_query(query), dtype=np.float32)

    def _semantic_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every document (vectors are normalised)."""
        q = self._embed_query(query)
        return self.doc_embeddings @ q  # (N,) cosine similarities in [-1, 1]

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> Tuple[List[Document], float]:
        """Retrieve top-k documents using hybrid (dense + BM25) scoring."""
        if self.doc_embeddings is None:
            raise ValueError("Vectorstore not loaded. Build or load it first.")

        start = time.time()
        order = self._hybrid_rank(query)[:k]
        results = [self.all_documents[i] for i in order]
        latency = time.time() - start
        self.retrieval_times.append(latency)
        return results, latency

    def retrieve_with_scores(
        self, query: str, k: int = TOP_K_RETRIEVAL
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """Retrieve top-k with a distance score (lower = closer), for the UI.

        Uses the same hybrid ranking as `retrieve` so the web UI and the
        inference pipeline agree on ordering. The returned score is
        ``1 - hybrid_score`` so existing callers can recover a similarity via
        ``1 - score``.

        Results with hybrid_score below MIN_CONFIDENCE_SCORE are filtered out
        to prevent low-confidence or hallucinated matches.
        """
        if self.doc_embeddings is None:
            raise ValueError("Vectorstore not loaded.")

        start = time.time()
        ranked = self._hybrid_rank(query, with_scores=True)
        # Filter by confidence threshold, then take top-k
        filtered = [(i, s) for i, s in ranked if s >= MIN_CONFIDENCE_SCORE][:k]
        results = [(self.all_documents[i], float(1.0 - s)) for i, s in filtered]

        # Optional cross-encoder reranking: re-scores (query, doc) pairs jointly
        if self._cross_encoder is not None and results:
            try:
                pairs = [(query, doc.page_content[:512]) for doc, _ in results]
                ce_scores = self._cross_encoder.predict(pairs)
                reranked = sorted(zip(results, ce_scores), key=lambda x: -x[1])
                results = [item for item, _ in reranked]
            except Exception as e:
                print(f"  Warning: cross-encoder reranking failed ({e}), using hybrid order.")

        latency = time.time() - start
        self.retrieval_times.append(latency)
        return results, latency

    def _hybrid_rank(self, query: str, with_scores: bool = False):
        """Return document indices ranked by the weighted hybrid score.

        If ``with_scores`` is True, returns ``[(idx, score), ...]`` instead of
        bare indices. Semantic cosine similarities are min-max normalised to
        [0, 1] so they share a scale with the normalised BM25 scores.
        """
        semantic = self._semantic_scores(query)
        sem_min, sem_max = float(semantic.min()), float(semantic.max())
        sem_norm = (
            (semantic - sem_min) / (sem_max - sem_min)
            if sem_max > sem_min
            else np.zeros_like(semantic)
        )

        bm25_scores = np.asarray(self.bm25.get_scores(query.lower().split()), dtype=np.float32)
        bm25_max = float(bm25_scores.max()) if bm25_scores.size else 0.0
        bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else np.zeros_like(bm25_scores)

        combined = SEMANTIC_WEIGHT * sem_norm + BM25_WEIGHT * bm25_norm
        order = np.argsort(-combined)
        if with_scores:
            return [(int(i), float(combined[i])) for i in order]
        return [int(i) for i in order]

    # ------------------------------------------------------------------ #
    # Helpers / stats                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def extract_standard_codes(self, documents: List[Document]) -> List[str]:
        """Extract unique standard codes from documents, preserving order."""
        seen, codes = set(), []
        for doc in documents:
            code = doc.metadata.get("standard_code", "")
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
        return codes

    def get_average_latency(self) -> float:
        if not self.retrieval_times:
            return 0.0
        return sum(self.retrieval_times) / len(self.retrieval_times)

    def reset_latency_tracking(self) -> None:
        self.retrieval_times = []

    def get_vectorstore_stats(self) -> Dict[str, Any]:
        if self.doc_embeddings is None:
            return {"status": "not_loaded"}
        return {
            "status": "loaded",
            "total_documents": len(self.all_documents),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": int(self.doc_embeddings.shape[1]),
            "persist_dir": self.persist_dir,
        }
