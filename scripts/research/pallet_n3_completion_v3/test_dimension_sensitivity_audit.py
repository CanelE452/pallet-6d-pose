"""Small aggregation tests for the pre-report dimension audit."""
from __future__ import annotations

import unittest

from .dimension_sensitivity_audit import summarize


class DimensionSensitivityAuditTests(unittest.TestCase):
    def test_six_passing_heads_are_required(self):
        fits = {f"fit{i}": {"status": "PASS", "changed_logit_entries": 1776}
                for i in range(6)}
        result = summarize(fits)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["fits_verified"], 6)
        self.assertEqual(result["total_changed_logit_entries"], 10656)

    def test_missing_or_unchanged_head_blocks_the_claim(self):
        fits = {f"fit{i}": {"status": "PASS", "changed_logit_entries": 1776}
                for i in range(5)}
        fits["fit5"] = {"status": "PASS", "changed_logit_entries": 0}
        result = summarize(fits)
        self.assertEqual(result["status"], "BLOCKED_INTEGRITY")
        self.assertFalse(result["all_six_changed"])


if __name__ == "__main__":
    unittest.main()
