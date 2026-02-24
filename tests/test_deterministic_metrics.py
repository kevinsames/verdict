"""Tests for deterministic metrics."""

import pytest
from verdict.evaluation.deterministic_metrics import (
    compute_rouge_l,
    compute_exact_match,
    compute_contains_match,
    compute_response_length,
    compute_token_count,
)


class TestRougeL:
    """Tests for ROUGE-L metric."""

    def test_identical_strings(self):
        """Identical strings should score 1.0."""
        result = compute_rouge_l("hello world", "hello world")
        assert result == 1.0

    def test_no_overlap(self):
        """No overlap should score 0.0."""
        result = compute_rouge_l("hello world", "foo bar")
        assert result == 0.0

    def test_partial_overlap(self):
        """Partial overlap should score between 0 and 1."""
        result = compute_rouge_l("the quick brown fox", "the quick fox")
        assert 0.0 < result < 1.0

    def test_case_insensitive(self):
        """ROUGE-L should be case-insensitive."""
        result = compute_rouge_l("Hello World", "hello world")
        assert result == 1.0

    def test_empty_strings(self):
        """Empty strings should return 0.0."""
        assert compute_rouge_l("", "test") == 0.0
        assert compute_rouge_l("test", "") == 0.0
        assert compute_rouge_l(None, "test") == 0.0
        assert compute_rouge_l("test", None) == 0.0


class TestExactMatch:
    """Tests for exact match metric."""

    def test_exact_match(self):
        """Identical strings should match."""
        assert compute_exact_match("hello", "hello") == 1.0

    def test_no_match(self):
        """Different strings should not match."""
        assert compute_exact_match("hello", "world") == 0.0

    def test_case_insensitive(self):
        """Match should be case-insensitive."""
        assert compute_exact_match("Hello", "HELLO") == 1.0

    def test_whitespace_trimmed(self):
        """Match should ignore leading/trailing whitespace."""
        assert compute_exact_match("  hello  ", "hello") == 1.0


class TestContainsMatch:
    """Tests for contains match metric."""

    def test_contains(self):
        """Response containing reference should match."""
        assert compute_contains_match("hello world", "hello") == 1.0

    def test_not_contains(self):
        """Response not containing reference should not match."""
        assert compute_contains_match("hello world", "foo") == 0.0

    def test_case_insensitive(self):
        """Contains should be case-insensitive."""
        assert compute_contains_match("Hello World", "hello") == 1.0


class TestResponseLength:
    """Tests for response length metric."""

    def test_basic_length(self):
        """Should return character count."""
        assert compute_response_length("hello") == 5

    def test_empty_string(self):
        """Empty string should return 0."""
        assert compute_response_length("") == 0

    def test_none(self):
        """None should return 0."""
        assert compute_response_length(None) == 0


class TestTokenCount:
    """Tests for token count metric."""

    def test_basic_count(self):
        """Should return word count."""
        assert compute_token_count("hello world") == 2

    def test_empty_string(self):
        """Empty string should return 0."""
        assert compute_token_count("") == 0

    def test_none(self):
        """None should return 0."""
        assert compute_token_count(None) == 0