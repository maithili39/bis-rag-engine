"""Vector retrieval system for BIS standards using free HuggingFace embeddings."""

import os
import time
import json
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
        self.bm25 = None  # BM25 ranker

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
            
            # Initialize BM25 for hybrid search
            self.all_documents = documents
            texts = [doc.page_content for doc in documents]
            tokenized_docs = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(tokenized_docs)
            
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
                
                # Reconstruct BM25 index from vectorstore
                try:
                    all_docs = self.vectorstore.get()
                    if all_docs and 'documents' in all_docs:
                        tokenized_docs = [text.lower().split() for text in all_docs['documents']]
                        self.bm25 = BM25Okapi(tokenized_docs)
                        self.all_documents = [Document(page_content=text, metadata=meta) 
                                            for text, meta in zip(all_docs['documents'], all_docs.get('metadatas', [{}]*len(all_docs['documents'])))]
                        print(f"  BM25 index reconstructed")
                except Exception as e:
                    print(f"  Warning: Could not reconstruct BM25 index: {e}")
                
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
            # Use hybrid search if BM25 is available
            if self.bm25 is not None:
                results = self._hybrid_retrieve(query, k=k)
            else:
                # Fallback to semantic search only
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                results = [doc for doc, _ in results]
            
            latency = time.time() - start_time
            self.retrieval_times.append(latency)
            
            return results, latency
        except Exception as e:
            print(f"  Error during retrieval: {e}")
            raise

    def _hybrid_retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """Hybrid retrieval combining semantic similarity and BM25 keyword matching."""
        # 1. Semantic search
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        semantic_docs = {i: (doc, score) for i, (doc, score) in enumerate(semantic_results)}
        
        # 2. BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 3. Normalize and combine scores
        combined_scores = {}
        
        # Add semantic scores
        for idx, (doc, sem_score) in semantic_docs.items():
            doc_idx = self._find_document_index(doc)
            if doc_idx is not None:
                combined_scores[doc_idx] = {"doc": doc, "semantic": 1.0 - sem_score, "bm25": 0.0}
        
        # Add BM25 scores
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 else 1.0
        for doc_idx, bm25_score in enumerate(bm25_scores):
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            if doc_idx not in combined_scores:
                combined_scores[doc_idx] = {"doc": self.all_documents[doc_idx], "semantic": 0.0, "bm25": normalized_bm25}
            else:
                combined_scores[doc_idx]["bm25"] = normalized_bm25
        
        # 4. Weighted combination - using optimized weights from config
        # For BIS standards: balance semantic understanding with keyword matching
        for doc_idx in combined_scores:
            sem = combined_scores[doc_idx]["semantic"]
            bm25 = combined_scores[doc_idx]["bm25"]
            combined_scores[doc_idx]["score"] = (SEMANTIC_WEIGHT * sem) + (BM25_WEIGHT * bm25)
        
        # 5. Sort and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        results = [item[1]["doc"] for item in sorted_results[:k]]
        
        return results

    def _find_document_index(self, doc: Document) -> Optional[int]:
        """Find the index of a document in the all_documents list."""
        for idx, stored_doc in enumerate(self.all_documents):
            if stored_doc.page_content == doc.page_content:
                return idx
        return None

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
