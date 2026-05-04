"""Vector retrieval system for BIS standards using free HuggingFace embeddings."""

import os
import time
import json
import pickle
import hashlib
import warnings
from typing import List, Tuple, Dict, Any, Optional

warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, TOP_K_RETRIEVAL, COLLECTION_NAME, SEMANTIC_WEIGHT, BM25_WEIGHT


class BISRetriever:
    """Retrieves relevant BIS standards using hybrid search (semantic + BM25 keyword).
    
    Uses free HuggingFace sentence-transformers for embeddings (no API key needed).
    Combines semantic similarity with BM25 keyword matching for better recall.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = None
        self.retrieval_times = []
        self.all_documents = []  # Store all docs for BM25
        self.bm25 = None         # BM25 ranker
        self._content_to_idx: Dict[str, int] = {}  # O(1) lookup map: content_hash -> index
        self._bm25_cache_path = os.path.join(persist_dir, "bm25_index.pkl")

    def build_vectorstore(self, documents: List[Document]) -> None:
        """Build the vector database from documents."""
        try:
            print(f"  Building vectorstore with {len(documents)} documents...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=COLLECTION_NAME
            )
            
            # Initialize BM25 and content-hash index
            self.all_documents = documents
            self._build_content_index(documents)
            texts = [doc.page_content for doc in documents]
            tokenized_docs = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(tokenized_docs)

            # Persist BM25 to disk so load_vectorstore() can skip re-fetching all docs
            self._save_bm25_cache(documents, tokenized_docs)

            print(f"  Vectorstore built and persisted to {self.persist_dir}")
            print(f"  BM25 index initialized with {len(documents)} documents")
        except Exception as e:
            print(f"  Error building vectorstore: {e}")
            raise

    def load_vectorstore(self) -> bool:
        """Load existing vectorstore from disk."""
        try:
            if not os.path.exists(self.persist_dir):
                return False
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            
            # Verify it has documents
            try:
                count = self.vectorstore._collection.count()
                if count == 0:
                    return False
                print(f"  Loaded vectorstore with {count} documents")
                
                # Fast path: load BM25 from disk cache (avoids re-fetching all docs from Chroma)
                if self._load_bm25_cache():
                    print(f"  BM25 index loaded from disk cache")
                else:
                    # Slow path: rebuild from ChromaDB (fallback)
                    print(f"  Rebuilding BM25 index from ChromaDB...")
                    all_docs = self.vectorstore.get()
                    if all_docs and 'documents' in all_docs:
                        texts = all_docs['documents']
                        metadatas = all_docs.get('metadatas', [{}] * len(texts))
                        self.all_documents = [
                            Document(page_content=t, metadata=m)
                            for t, m in zip(texts, metadatas)
                        ]
                        self._build_content_index(self.all_documents)
                        tokenized_docs = [t.lower().split() for t in texts]
                        self.bm25 = BM25Okapi(tokenized_docs)
                        # Save cache for next time
                        self._save_bm25_cache(self.all_documents, tokenized_docs)
                        print(f"  BM25 index rebuilt and cached")
                
                return True
            except:
                return False
                
        except Exception as e:
            print(f"  Error loading vectorstore: {e}")
            return False

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> Tuple[List[Document], float]:
        """Retrieve top-k relevant documents using hybrid search (semantic + BM25)."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not loaded. Build or load vectorstore first.")
        
        start_time = time.time()
        try:
            if self.bm25 is not None:
                results = self._hybrid_retrieve(query, k=k)
            else:
                results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
                results = [doc for doc, _ in results_with_scores]
            
            latency = time.time() - start_time
            self.retrieval_times.append(latency)
            return results, latency
        except Exception as e:
            print(f"  Error during retrieval: {e}")
            raise

    def _hybrid_retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """Hybrid retrieval combining semantic similarity and BM25 keyword matching."""
        # 1. Semantic search — fetch k candidates (not k*2, since k is already 12)
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # 2. BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 3. Normalize and combine scores using O(1) hash-map lookup
        combined_scores: Dict[int, Dict] = {}
        
        # Add semantic scores
        for doc, sem_score in semantic_results:
            doc_idx = self._content_to_idx.get(self._hash(doc.page_content))
            if doc_idx is not None:
                combined_scores[doc_idx] = {
                    "doc": doc,
                    "semantic": 1.0 - sem_score,
                    "bm25": 0.0
                }
        
        # Add BM25 scores
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 else 1.0
        for doc_idx, bm25_score in enumerate(bm25_scores):
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            if doc_idx not in combined_scores:
                combined_scores[doc_idx] = {
                    "doc": self.all_documents[doc_idx],
                    "semantic": 0.0,
                    "bm25": normalized_bm25
                }
            else:
                combined_scores[doc_idx]["bm25"] = normalized_bm25
        
        # 4. Weighted combination
        for entry in combined_scores.values():
            entry["score"] = (SEMANTIC_WEIGHT * entry["semantic"]) + (BM25_WEIGHT * entry["bm25"])
        
        # 5. Sort and return top k
        sorted_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        return [entry["doc"] for entry in sorted_results[:k]]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash(text: str) -> str:
        """Fast content fingerprint for O(1) lookup."""
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def _build_content_index(self, documents: List[Document]) -> None:
        """Build hash-map from content fingerprint to list index."""
        self._content_to_idx = {
            self._hash(doc.page_content): idx
            for idx, doc in enumerate(documents)
        }

    def _save_bm25_cache(self, documents: List[Document], tokenized_docs: List[List[str]]) -> None:
        """Persist BM25 index and documents to disk to avoid rebuild on next load."""
        try:
            cache = {
                "documents": [(doc.page_content, doc.metadata) for doc in documents],
                "bm25": self.bm25,
            }
            with open(self._bm25_cache_path, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Warning: Could not save BM25 cache: {e}")

    def _load_bm25_cache(self) -> bool:
        """Load BM25 index from disk cache. Returns True on success."""
        if not os.path.exists(self._bm25_cache_path):
            return False
        try:
            with open(self._bm25_cache_path, "rb") as f:
                cache = pickle.load(f)
            self.all_documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in cache["documents"]
            ]
            self.bm25 = cache["bm25"]
            self._build_content_index(self.all_documents)
            return True
        except Exception as e:
            print(f"  Warning: Could not load BM25 cache: {e}")
            return False

    def retrieve_with_scores(self, query: str, k: int = TOP_K_RETRIEVAL) -> Tuple[List[Tuple[Document, float]], float]:
        """Retrieve top-k standards with relevance scores."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not loaded.")
        
        start_time = time.time()
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        latency = time.time() - start_time
        self.retrieval_times.append(latency)
        return results, latency

    def extract_standard_codes(self, documents: List[Document]) -> List[str]:
        """Extract unique standard codes from retrieved documents, preserving order."""
        seen = set()
        codes = []
        
        for doc in documents:
            code = doc.metadata.get("standard_code", "")
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
        
        return codes

    def get_average_latency(self) -> float:
        """Get average retrieval latency."""
        if not self.retrieval_times:
            return 0.0
        return sum(self.retrieval_times) / len(self.retrieval_times)

    def reset_latency_tracking(self) -> None:
        """Reset latency tracking."""
        self.retrieval_times = []

    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the vectorstore."""
        if not self.vectorstore:
            return {"status": "not_loaded"}
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "status": "loaded",
                "total_documents": count,
                "embedding_model": EMBEDDING_MODEL,
                "persist_dir": self.persist_dir
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
