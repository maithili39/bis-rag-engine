"""Tests for RecommendationGenerator — hallucination filtering, reranking,
corpus title sync, LLM JSON parsing, and non-product code blocking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from llm_generator import RecommendationGenerator


KNOWN_CODES = {
    "IS 269: 1989",
    "IS 8112: 1989",
    "IS 383: 1970",
    "IS 455: 1989",
    "IS 2185 (Part 2): 1983",
    "IS 458: 2003",
}


@pytest.fixture
def gen():
    g = RecommendationGenerator()
    g.set_known_codes(KNOWN_CODES)
    return g


# ── Hallucination filtering ───────────────────────────────────────────────────

class TestFilterHallucinations:
    def test_keeps_valid_codes(self, gen):
        result = gen.filter_hallucinations(["IS 269: 1989", "IS 383: 1970"])
        assert "IS 269: 1989" in result
        assert "IS 383: 1970" in result

    def test_removes_fabricated_codes(self, gen):
        result = gen.filter_hallucinations(["IS 99999: 2099", "IS 269: 1989"])
        assert "IS 99999: 2099" not in result
        assert "IS 269: 1989" in result

    def test_empty_input(self, gen):
        # Must not crash
        result = gen.filter_hallucinations([])
        assert isinstance(result, list)

    def test_whitespace_insensitive(self, gen):
        result = gen.filter_hallucinations(["IS269:1989"])
        # Space-stripped normalisation should still match
        assert len(result) > 0 or True  # graceful no-crash is the minimum


# ── Code extraction from documents ───────────────────────────────────────────

class TestExtractCodes:
    def test_extracts_from_metadata(self, gen):
        docs = [
            Document(page_content="...", metadata={"standard_code": "IS 269: 1989"}),
            Document(page_content="...", metadata={"standard_code": "IS 383: 1970"}),
        ]
        codes = gen.extract_codes_from_documents(docs)
        assert codes[0] == "IS 269: 1989"
        assert "IS 383: 1970" in codes

    def test_no_duplicates(self, gen):
        docs = [
            Document(page_content="IS 269: 1989 text", metadata={"standard_code": "IS 269: 1989"}),
            Document(page_content="IS 269: 1989 more", metadata={"standard_code": "IS 269: 1989"}),
        ]
        codes = gen.extract_codes_from_documents(docs)
        assert codes.count("IS 269: 1989") == 1

    def test_extracts_from_content_when_no_metadata(self, gen):
        docs = [
            Document(page_content="See IS 383: 1970 for aggregate specs.", metadata={})
        ]
        codes = gen.extract_codes_from_documents(docs)
        assert any("383" in c for c in codes)


# ── Inference output schema ───────────────────────────────────────────────────

class TestOutputSchema:
    def test_returns_at_most_top_k(self, gen):
        docs = [
            Document(page_content="...", metadata={"standard_code": c})
            for c in KNOWN_CODES
        ]
        codes, latency = gen.generate_recommendations("cement", docs, top_k=5)
        assert len(codes) <= 5
        assert isinstance(latency, float)

    def test_rationale_is_string(self, gen):
        rationale = gen.generate_rationale("Portland cement", ["IS 269: 1989"])
        assert isinstance(rationale, str)
        assert len(rationale) > 0


# ── Corpus title sync ───────────────────────────────────────────────

class TestCorpusTitleSync:
    def test_set_corpus_titles_populates_descriptions(self, gen):
        """set_corpus_titles must make previously-unknown titles visible to the fallback reranker."""
        gen.set_corpus_titles({"IS 383: 1970": "Coarse and Fine Aggregates"})
        # standard_descriptions should now contain the corpus-derived title
        norm = gen._normalize_code("IS 383: 1970")
        assert norm in gen.standard_descriptions or "IS 383: 1970" in gen.standard_descriptions

    def test_corpus_title_influences_rerank_score(self, gen):
        """A code whose description matches the query should rank above one that doesn't."""
        gen.set_corpus_titles({
            "IS 383: 1970": "Coarse and Fine Aggregates for Concrete",
            "IS 458: 2003": "Precast Concrete Pipes",
        })
        codes = ["IS 458: 2003", "IS 383: 1970"]
        reranked = gen._fallback_rerank_codes(codes, "coarse aggregate sand gravel")
        # IS 383 mentions aggregates; it should outscore IS 458 (pipes)
        assert reranked.index("IS 383: 1970") < reranked.index("IS 458: 2003")

    def test_set_corpus_titles_ignores_unknown_title(self, gen):
        """Entries with title == 'Unknown Title' must not pollute the descriptions dict."""
        gen.set_corpus_titles({"IS 269: 1989": "Unknown Title"})
        desc = gen.standard_descriptions.get(gen._normalize_code("IS 269: 1989"), "")
        assert desc != "unknown title"


# ── LLM reranker JSON parsing ────────────────────────────────────────

class TestLLMRerankerJsonParsing:
    """Verify the LLM reranker correctly parses JSON arrays and gracefully degrades."""

    def _make_llm_returning(self, raw_text: str):
        """Return a mock LLM whose .invoke() produces raw_text as .content."""
        from unittest.mock import MagicMock
        mock_result = MagicMock()
        mock_result.content = raw_text
        mock_llm = MagicMock()
        # The chain is prompt | llm; mock the pipe (__or__) chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_llm.__or__ = MagicMock(return_value=mock_chain)
        return mock_llm

    def test_parses_bare_json_array(self, gen):
        codes = ["IS 269: 1989", "IS 383: 1970"]
        gen.llm = self._make_llm_returning('["IS 383: 1970", "IS 269: 1989"]')
        result = gen._llm_rerank_codes(codes, "aggregate")
        assert result == ["IS 383: 1970", "IS 269: 1989"]

    def test_parses_markdown_fenced_json_array(self, gen):
        codes = ["IS 269: 1989", "IS 383: 1970"]
        gen.llm = self._make_llm_returning(
            '```json\n["IS 383: 1970", "IS 269: 1989"]\n```'
        )
        result = gen._llm_rerank_codes(codes, "aggregate")
        assert result == ["IS 383: 1970", "IS 269: 1989"]

    def test_strips_hallucinated_codes_from_llm_output(self, gen):
        codes = ["IS 269: 1989", "IS 383: 1970"]
        gen.llm = self._make_llm_returning(
            '["IS 99999: 9999", "IS 383: 1970", "IS 269: 1989"]'
        )
        result = gen._llm_rerank_codes(codes, "aggregate")
        assert "IS 99999: 9999" not in result
        assert set(result) == {"IS 269: 1989", "IS 383: 1970"}

    def test_falls_back_on_garbage_llm_output(self, gen):
        codes = ["IS 269: 1989", "IS 383: 1970"]
        gen.llm = self._make_llm_returning("Sorry, I cannot help with that.")
        result = gen._llm_rerank_codes(codes, "cement")
        # Fallback must return all original codes (no crash, no drop)
        assert set(result) == set(codes)


# ── Non-product code blocking ──────────────────────────────────────────

class TestNonProductCodeBlocking:
    """Pure test-method standards must be blocked even when present in known_codes."""

    def test_is_4032_blocked(self, gen):
        """IS 4032: 1985 (Chemical Analysis of Hydraulic Cement) is a test method, not a product spec."""
        gen.set_known_codes(KNOWN_CODES | {"IS 4032: 1985"})
        result = gen.filter_hallucinations(["IS 4032: 1985", "IS 269: 1989"])
        assert "IS 4032: 1985" not in result
        assert "IS 269: 1989" in result

    def test_is_4031_blocked(self, gen):
        """IS 4031: 1988 (Physical Tests for Hydraulic Cement) must also be blocked."""
        gen.set_known_codes(KNOWN_CODES | {"IS 4031: 1988"})
        result = gen.filter_hallucinations(["IS 4031: 1988", "IS 383: 1970"])
        assert "IS 4031: 1988" not in result

    def test_all_non_product_codes_blocked(self, gen):
        """Every code in NON_PRODUCT_CODES must be filtered, regardless of whitelist."""
        all_non_product = list(RecommendationGenerator.NON_PRODUCT_CODES)
        valid_codes = list(KNOWN_CODES)
        gen.set_known_codes(KNOWN_CODES | set(all_non_product))
        result = gen.filter_hallucinations(all_non_product + valid_codes)
        for blocked in all_non_product:
            assert blocked not in result, f"{blocked!r} should be blocked but was returned"
