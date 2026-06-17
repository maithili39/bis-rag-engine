"""Tests for BISRetriever — vectorstore persistence, version guard, retrieval.

Embeddings are mocked so these tests run without downloading any model.
Mark with ``pytest -m slow`` to skip in fast CI (they do hit disk I/O).
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from retriever import BISRetriever
from config import EMBEDDING_MODEL, MIN_CONFIDENCE_SCORE


# ── Helpers ───────────────────────────────────────────────────────────────────

_N_DIM = 8
_DOCS = [
    Document(
        page_content="BIS Standard: IS 269: 1989\nTitle: Ordinary Portland Cement\nScope: OPC 33 grade.",
        metadata={"standard_code": "IS 269: 1989", "title": "Ordinary Portland Cement", "source": "test"},
    ),
    Document(
        page_content="BIS Standard: IS 383: 1970\nTitle: Coarse and Fine Aggregates\nScope: Aggregates for concrete.",
        metadata={"standard_code": "IS 383: 1970", "title": "Coarse and Fine Aggregates", "source": "test"},
    ),
    Document(
        page_content="BIS Standard: IS 455: 1989\nTitle: Portland Slag Cement\nScope: PSC specification.",
        metadata={"standard_code": "IS 455: 1989", "title": "Portland Slag Cement", "source": "test"},
    ),
]


def _make_mock_embeddings():
    """Return deterministic mock embeddings.

    Doc i gets a unit vector with 1.0 at index i (modulo N_DIM).
    Query always returns a unit vector with 1.0 at index 0 (most similar to doc 0).
    All vectors are L2-normalised so cosine similarity == dot product.
    """
    mock = MagicMock()

    def embed_documents(texts):
        vecs = []
        for i in range(len(texts)):
            v = np.zeros(_N_DIM, dtype=np.float32)
            v[i % _N_DIM] = 1.0
            vecs.append(v.tolist())
        return vecs

    def embed_query(text):
        v = np.zeros(_N_DIM, dtype=np.float32)
        v[0] = 1.0
        return v.tolist()

    mock.embed_documents.side_effect = embed_documents
    mock.embed_query.side_effect = embed_query
    return mock


@pytest.fixture
def retriever(tmp_path):
    """A BISRetriever with mocked embeddings backed by tmp_path."""
    with patch("retriever.HuggingFaceEmbeddings", return_value=_make_mock_embeddings()):
        r = BISRetriever(persist_dir=str(tmp_path))
    return r


@pytest.fixture
def retriever_with_store(retriever, tmp_path):
    """A BISRetriever that has already built and loaded a vectorstore."""
    retriever.build_vectorstore(_DOCS)
    return retriever


# ── Build / persist / load round-trip ─────────────────────────────────────────

class TestVectorstoreRoundTrip:
    def test_build_creates_files(self, retriever_with_store, tmp_path):
        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "store.json").exists()

    def test_load_restores_document_count(self, retriever, tmp_path):
        retriever.build_vectorstore(_DOCS)
        # Fresh retriever, same path
        with patch("retriever.HuggingFaceEmbeddings", return_value=_make_mock_embeddings()):
            r2 = BISRetriever(persist_dir=str(tmp_path))
        loaded = r2.load_vectorstore()
        assert loaded is True
        assert len(r2.all_documents) == len(_DOCS)

    def test_load_restores_doc_content(self, retriever, tmp_path):
        retriever.build_vectorstore(_DOCS)
        with patch("retriever.HuggingFaceEmbeddings", return_value=_make_mock_embeddings()):
            r2 = BISRetriever(persist_dir=str(tmp_path))
        r2.load_vectorstore()
        original_codes = {d.metadata["standard_code"] for d in _DOCS}
        loaded_codes = {d.metadata.get("standard_code") for d in r2.all_documents}
        assert original_codes == loaded_codes

    def test_embeddings_matrix_shape_preserved(self, retriever, tmp_path):
        retriever.build_vectorstore(_DOCS)
        with patch("retriever.HuggingFaceEmbeddings", return_value=_make_mock_embeddings()):
            r2 = BISRetriever(persist_dir=str(tmp_path))
        r2.load_vectorstore()
        assert r2.doc_embeddings is not None
        assert r2.doc_embeddings.shape == (len(_DOCS), _N_DIM)

    def test_load_returns_false_when_no_store(self, retriever):
        """Calling load before build must return False, not raise."""
        result = retriever.load_vectorstore()
        assert result is False


# ── Embedding model version guard ─────────────────────────────────────────────

class TestModelVersionGuard:
    def test_rejects_stale_store(self, retriever_with_store, tmp_path):
        """A store built with a different model name must not load silently."""
        store_path = tmp_path / "store.json"
        with open(store_path, encoding="utf-8") as f:
            store = json.load(f)
        store["embedding_model"] = "wrong-model/different-than-config"
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(store, f)

        with patch("retriever.HuggingFaceEmbeddings", return_value=_make_mock_embeddings()):
            r2 = BISRetriever(persist_dir=str(tmp_path))
        result = r2.load_vectorstore()
        assert result is False, (
            "load_vectorstore() should return False when embedding model name "
            "in store.json doesn't match the configured EMBEDDING_MODEL."
        )

    def test_accepts_matching_model(self, retriever, tmp_path):
        """A store built with the current model name must load successfully."""
        retriever.build_vectorstore(_DOCS)
        with patch("retriever.HuggingFaceEmbeddings", return_value=_make_mock_embeddings()):
            r2 = BISRetriever(persist_dir=str(tmp_path))
        assert r2.load_vectorstore() is True


# ── Retrieval ─────────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_returns_documents(self, retriever_with_store):
        docs, latency = retriever_with_store.retrieve("cement", k=2)
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)
        assert len(docs) <= 2

    def test_respects_k(self, retriever_with_store):
        docs, _ = retriever_with_store.retrieve("cement", k=1)
        assert len(docs) == 1

    def test_latency_is_positive(self, retriever_with_store):
        _, latency = retriever_with_store.retrieve("cement", k=2)
        assert latency > 0

    def test_retrieve_raises_before_build(self, retriever):
        with pytest.raises(ValueError, match="not loaded"):
            retriever.retrieve("cement")

    def test_retrieve_with_scores_returns_tuples(self, retriever_with_store):
        results, latency = retriever_with_store.retrieve_with_scores("cement", k=2)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)


# ── Code extraction helper ────────────────────────────────────────────────────

class TestExtractStandardCodes:
    def test_extracts_unique_codes(self, retriever_with_store):
        codes = retriever_with_store.extract_standard_codes(_DOCS)
        assert len(codes) == len(set(codes)), "extract_standard_codes returned duplicates"
        assert "IS 269: 1989" in codes

    def test_empty_metadata_skipped(self, retriever_with_store):
        docs = [Document(page_content="no code here", metadata={})]
        codes = retriever_with_store.extract_standard_codes(docs)
        assert codes == []


# ── Stats ─────────────────────────────────────────────────────────────────────

class TestVectorstoreStats:
    def test_stats_before_load(self, retriever):
        stats = retriever.get_vectorstore_stats()
        assert stats["status"] == "not_loaded"

    def test_stats_after_build(self, retriever_with_store):
        stats = retriever_with_store.get_vectorstore_stats()
        assert stats["status"] == "loaded"
        assert stats["total_documents"] == len(_DOCS)
        assert "embedding_model" in stats
        assert "embedding_dim" in stats
        assert stats["embedding_dim"] == _N_DIM

    def test_average_latency_after_retrieve(self, retriever_with_store):
        assert retriever_with_store.get_average_latency() == 0.0
        retriever_with_store.retrieve("cement", k=2)
        assert retriever_with_store.get_average_latency() > 0

    def test_reset_clears_latency(self, retriever_with_store):
        retriever_with_store.retrieve("cement", k=2)
        retriever_with_store.reset_latency_tracking()
        assert retriever_with_store.get_average_latency() == 0.0
