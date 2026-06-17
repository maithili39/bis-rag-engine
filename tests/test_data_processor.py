"""Tests for BISDataProcessor — parsing, normalisation, chunking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from data_processor import BISDataProcessor


@pytest.fixture
def processor():
    return BISDataProcessor()


# ── Code formatting / normalisation ──────────────────────────────────────────

class TestFormatCode:
    def test_simple(self, processor):
        assert processor._format_code("269", "", "1989") == "IS 269: 1989"

    def test_part_with_number(self, processor):
        code = processor._format_code("2185", "2", "1983")
        assert code == "IS 2185 (Part 2): 1983"
        assert "Part Part" not in code

    def test_part_info_already_contains_part(self, processor):
        # Simulate a PDF where the capture group already reads "Part 2"
        code = processor._format_code("2185", "Part 2", "1983")
        assert code == "IS 2185 (Part 2): 1983"
        assert "Part Part" not in code

    def test_part_info_uppercase(self, processor):
        code = processor._format_code("2185", "PART 3", "1983")
        assert "Part Part" not in code
        assert "2185" in code

    def test_normalize_extra_spaces(self, processor):
        code = processor._normalize_code("IS  269 :  1989")
        assert code == "IS 269: 1989"

    def test_normalize_colon_spacing(self, processor):
        code = processor._normalize_code("IS 383:1970")
        assert code == "IS 383: 1970"


# ── Known Codes Whitelist ─────────────────────────────────────────────────────

class TestKnownCodesJson:
    import json
    import re
    from pathlib import Path
    
    # Get the actual file path
    KNOWN_CODES_PATH = Path(__file__).parent.parent / "known_codes.json"
    
    @pytest.fixture
    def known_codes(self):
        if not self.KNOWN_CODES_PATH.exists():
            pytest.skip("known_codes.json not found")
        with open(self.KNOWN_CODES_PATH, "r", encoding="utf-8") as f:
            return self.json.load(f)
            
    def test_no_part_part_in_whitelist(self, known_codes):
        for code in known_codes:
            assert "Part Part" not in code, f"Corrupted entry found: {code}"
            
    def test_no_duplicate_normalized_codes(self, processor, known_codes):
        normalized = [processor._normalize_code(c) for c in known_codes]
        duplicates = set(x for x in normalized if normalized.count(x) > 1)
        assert not duplicates, f"Duplicate normalized entries found: {duplicates}"
        
    def test_all_codes_valid_format(self, known_codes):
        pattern = self.re.compile(r'^IS\s+\d+.*:\s*\d{4}$')
        for code in known_codes:
            assert pattern.match(code), f"Invalid code format: {code}"



# ── Standard section parsing ──────────────────────────────────────────────────

class TestParseStandardSection:
    def test_basic_section(self, processor):
        section = (
            "SUMMARY OF\n"
            "IS 269 : 1989  ORDINARY PORTLAND CEMENT\n"
            "1. Scope — This standard covers ordinary portland cement."
        )
        result = processor._parse_standard_section(section)
        assert result is not None
        assert result["code"] == "IS 269: 1989"
        assert "cement" in result["content"].lower()

    def test_returns_none_for_short_section(self, processor):
        assert processor._parse_standard_section("too short") is None

    def test_returns_none_for_no_is_code(self, processor):
        section = "Some section without any IS code in it.\n" * 5
        assert processor._parse_standard_section(section) is None


# ── Document creation ─────────────────────────────────────────────────────────

class TestCreateDocuments:
    def test_metadata_preserved(self, processor):
        standards = [
            {"code": "IS 269: 1989", "title": "Ordinary Portland Cement",
             "content": "Scope: covers OPC.", "full_text": "..."}
        ]
        docs = processor.create_documents_from_standards(standards)
        assert len(docs) == 1
        assert docs[0].metadata["standard_code"] == "IS 269: 1989"
        assert "IS 269: 1989" in docs[0].page_content

    def test_chunking_preserves_code_in_metadata(self, processor):
        standards = [
            {"code": "IS 383: 1970", "title": "Aggregates",
             "content": "x " * 600, "full_text": "..."}
        ]
        docs = processor.create_documents_from_standards(standards)
        chunks = processor.chunk_documents(docs)
        for chunk in chunks:
            # Every chunk must carry the parent standard's code
            assert chunk.metadata.get("standard_code") == "IS 383: 1970"


# ── Parsing Coverage ─────────────────────────────────────────────────────────

class TestParsingCoverage:
    def test_coverage_calculation(self, processor):
        processor.all_standard_codes = {"IS 269: 1989", "IS 383: 1970", "IS 455: 1989"}
        toc = ["IS 269: 1989", "IS 383: 1970", "IS 458: 2003"]
        coverage = processor.compute_parsing_coverage(toc)
        # 2 out of 3 found
        assert coverage == 2.0 / 3.0
        
    def test_empty_toc_returns_zero(self, processor):
        assert processor.compute_parsing_coverage([]) == 0.0
