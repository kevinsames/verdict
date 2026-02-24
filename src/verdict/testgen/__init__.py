"""
RAG Test Dataset Generator module for Verdict.

Generates synthetic test datasets from Qdrant vector database for RAG evaluation.
"""

from verdict.testgen.config import Settings
from verdict.testgen.generate_dataset import TestDatasetGenerator

__all__ = [
    "Settings",
    "TestDatasetGenerator",
]
