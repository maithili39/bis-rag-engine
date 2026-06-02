"""Verify the inference.py output schema strictly matches eval_script.py expectations."""
import json, sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

# Load the public test input
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_INPUT = os.path.join(DATA_DIR, "public_test_set.json")
TEST_RESULTS = os.path.join(DATA_DIR, "public_test_results.json")


@pytest.mark.skipif(not os.path.exists(TEST_RESULTS), reason="pre-computed results not present")
class TestInferenceOutputSchema:
    """Schema tests run against the committed public_test_results.json."""

    @pytest.fixture(scope="class")
    def results(self):
        with open(TEST_RESULTS) as f:
            return json.load(f)

    def test_is_list(self, results):
        assert isinstance(results, list)

    def test_required_keys_present(self, results):
        required = {"id", "query", "retrieved_standards", "latency_seconds"}
        for item in results:
            missing = required - item.keys()
            assert not missing, f"{item['id']} missing keys: {missing}"

    def test_retrieved_standards_is_list(self, results):
        for item in results:
            assert isinstance(item["retrieved_standards"], list), item["id"]

    def test_retrieved_standards_count(self, results):
        for item in results:
            count = len(item["retrieved_standards"])
            assert count <= 5, f"{item['id']} returned {count} > 5 standards"

    def test_latency_is_positive_float(self, results):
        for item in results:
            lat = item["latency_seconds"]
            assert isinstance(lat, (int, float)) and lat >= 0, item["id"]

    def test_standard_code_format(self, results):
        import re
        pattern = re.compile(r"IS\s+\d+", re.IGNORECASE)
        for item in results:
            for code in item["retrieved_standards"]:
                assert pattern.match(code), f"Bad code format in {item['id']}: {code!r}"

    def test_no_part_part_in_codes(self, results):
        for item in results:
            for code in item["retrieved_standards"]:
                assert "Part Part" not in code, f"'Part Part' found: {code!r}"


@pytest.mark.skipif(not os.path.exists(TEST_RESULTS), reason="pre-computed results not present")
class TestEvalMetrics:
    """Sanity-check that the saved results achieve reasonable metrics."""

    def _compute_metrics(self, results):
        def norm(s):
            return str(s).replace(" ", "").lower()

        hits3 = 0
        mrr_sum = 0.0
        for item in results:
            expected = {norm(s) for s in item.get("expected_standards", [])}
            retrieved = [norm(s) for s in item.get("retrieved_standards", [])]
            if any(s in expected for s in retrieved[:3]):
                hits3 += 1
            for rank, s in enumerate(retrieved[:5], 1):
                if s in expected:
                    mrr_sum += 1.0 / rank
                    break
        n = len(results)
        return hits3 / n * 100, mrr_sum / n

    def test_hit_rate_above_50_percent(self):
        with open(TEST_RESULTS) as f:
            results = json.load(f)
        hit_rate, _ = self._compute_metrics(results)
        assert hit_rate >= 50.0, f"Hit Rate@3 is only {hit_rate:.1f}% — pipeline may be broken"

    def test_mrr_above_0_4(self):
        with open(TEST_RESULTS) as f:
            results = json.load(f)
        _, mrr = self._compute_metrics(results)
        assert mrr >= 0.4, f"MRR@5 is only {mrr:.3f} — pipeline may be broken"
