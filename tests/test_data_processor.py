"""Tests for BISDataProcessor — parsing, normalisation, chunking."""
import sys, os, re
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

    # Regression tests: pypdf sometimes drops the space between "Part" and the
    # number/letter that follows (e.g. "(PART11)" instead of "(Part 11)"). A
    # `\bpart\b`-based guard doesn't catch this because \b never matches between
    # two word characters ("t" and "1" are both \w), so these slipped through
    # and got double-prefixed to "Part PART11". The guard must use a plain
    # startswith check instead.
    def test_part_info_no_space_uppercase(self, processor):
        code = processor._format_code("10124", "PART11", "1988")
        assert code == "IS 10124 (PART11): 1988"
        assert len(re.findall(r"part", code, re.IGNORECASE)) == 1

    def test_part_info_no_space_titlecase(self, processor):
        code = processor._format_code("1367", "Part1", "2002")
        assert code == "IS 1367 (Part1): 2002"
        assert len(re.findall(r"part", code, re.IGNORECASE)) == 1

    def test_part_info_roman_numeral_no_space(self, processor):
        code = processor._format_code("432", "PARTII", "1982")
        assert len(re.findall(r"part", code, re.IGNORECASE)) == 1

    def test_part_info_plural_parts(self, processor):
        # "(Parts 1 to 10)" must not become "Part Parts 1 to 10"
        code = processor._format_code("191", "Parts 1 to 10", "1980")
        assert "IS 191 (Parts 1 to 10): 1980" == code
        assert len(re.findall(r"part", code, re.IGNORECASE)) == 1


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


# ── known_codes.json whitelist integrity ─────────────────────────────────────

KNOWN_CODES_PATH = os.path.join(os.path.dirname(__file__), "..", "known_codes.json")


@pytest.mark.skipif(not os.path.exists(KNOWN_CODES_PATH), reason="known_codes.json not built yet")
class TestKnownCodesJson:
    """Validate the committed known_codes.json whitelist has no data corruption.

    These tests FAIL on the current stale artifacts (built before the
    _format_code startswith fix) and PASS after the vectorstore is rebuilt.
    That is the intended behaviour — the failures surface the corruption.
    """

    @pytest.fixture(scope="class")
    def codes(self):
        with open(KNOWN_CODES_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_no_part_part_in_whitelist(self, codes):
        bad = [c for c in codes if "Part Part" in c or "PART Part" in c]
        assert not bad, (
            f"'Part Part' double-prefix corruption found in known_codes.json:\n"
            + "\n".join(bad)
        )

    def test_no_duplicate_normalized_codes(self, codes):
        """Detects pairs like ('IS 10124 (PART11): 1988', 'IS 10124 (Part 11): 1988').
        After a clean rebuild these normalize to the same string — duplicates are gone.
        """
        import collections

        def _norm(c):
            return c.replace(" ", "").lower()

        counts = collections.Counter(_norm(c) for c in codes)
        dupes = {k: v for k, v in counts.items() if v > 1}
        if dupes:
            # Build human-readable list: show which original entries collapsed
            detail = []
            for norm_key in sorted(dupes):
                originals = [c for c in codes if _norm(c) == norm_key]
                detail.append(f"  {norm_key!r}: {originals}")
            pytest.fail(
                f"{len(dupes)} duplicate normalized code(s) found in known_codes.json "
                f"(rebuild vectorstore to fix):\n" + "\n".join(detail)
            )

    def test_all_codes_valid_format(self, codes):
        bad = [c for c in codes if not re.match(r"IS\s+\d+", c, re.IGNORECASE)]
        assert not bad, f"Non-IS-format entries in known_codes.json: {bad[:10]}"
