"""Tests for regression detector."""

import pytest
from verdict.regression.regression_detector import (
    RegressionDetector,
    VerdictLabel,
)


class TestVerdictLabel:
    """Tests for VerdictLabel enum."""

    def test_values(self):
        """VerdictLabel should have correct values."""
        assert VerdictLabel.PASS.value == "PASS"
        assert VerdictLabel.WARN.value == "WARN"
        assert VerdictLabel.FAIL.value == "FAIL"


class TestRegressionDetector:
    """Tests for RegressionDetector class."""

    def test_init(self):
        """Detector should initialize with defaults."""
        detector = RegressionDetector()
        assert detector.threshold_pct == 5.0
        assert detector.p_value_threshold == 0.05

    def test_init_custom_params(self):
        """Detector should accept custom parameters."""
        detector = RegressionDetector(
            threshold_pct=10.0,
            p_value_threshold=0.01
        )
        assert detector.threshold_pct == 10.0
        assert detector.p_value_threshold == 0.01

    def test_determine_verdict_pass(self):
        """No regressions should return PASS."""
        detector = RegressionDetector()
        comparisons = [
            {"metric_name": "faithfulness", "is_regression": False},
            {"metric_name": "answer_relevance", "is_regression": False},
        ]
        verdict = detector._determine_verdict(comparisons)
        assert verdict == VerdictLabel.PASS

    def test_determine_verdict_warn(self):
        """Minor regression should return WARN."""
        detector = RegressionDetector(threshold_pct=5.0)
        comparisons = [
            {"metric_name": "faithfulness", "is_regression": True, "pct_change": -6.0},
        ]
        verdict = detector._determine_verdict(comparisons)
        assert verdict == VerdictLabel.WARN

    def test_determine_verdict_fail(self):
        """Severe regression should return FAIL."""
        detector = RegressionDetector(threshold_pct=5.0)
        comparisons = [
            {"metric_name": "faithfulness", "is_regression": True, "pct_change": -15.0},
        ]
        verdict = detector._determine_verdict(comparisons)
        assert verdict == VerdictLabel.FAIL