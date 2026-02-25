"""
Deterministic metrics for Verdict.

Implements ROUGE-L, exact match, response length, and latency metrics
that don't require LLM evaluation.
"""

import logging
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rouge_l(response: Optional[str], reference: Optional[str]) -> float:
    """
    Compute ROUGE-L score between response and reference.

    ROUGE-L uses longest common subsequence (LCS) to measure similarity.

    Args:
        response: Generated response text.
        reference: Ground truth reference text.

    Returns:
        ROUGE-L F1 score between 0 and 1.
    """
    if not response or not reference:
        return 0.0

    # Tokenize (simple whitespace)
    response_tokens = response.lower().split()
    reference_tokens = reference.lower().split()

    if not response_tokens or not reference_tokens:
        return 0.0

    # Compute LCS length
    lcs_length = _lcs_length(response_tokens, reference_tokens)

    # Compute precision and recall
    precision = lcs_length / len(response_tokens) if response_tokens else 0.0
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0.0

    # F1 score
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(seq1: list, seq2: list) -> int:
    """
    Compute length of longest common subsequence.

    Args:
        seq1: First sequence.
        seq2: Second sequence.

    Returns:
        Length of LCS.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_exact_match(response: Optional[str], reference: Optional[str]) -> float:
    """
    Compute exact match score.

    Args:
        response: Generated response text.
        reference: Ground truth reference text.

    Returns:
        1.0 if exact match (case-insensitive, trimmed), 0.0 otherwise.
    """
    if not response or not reference:
        return 0.0

    return 1.0 if response.strip().lower() == reference.strip().lower() else 0.0


def compute_contains_match(response: Optional[str], reference: Optional[str]) -> float:
    """
    Compute whether response contains the reference.

    Args:
        response: Generated response text.
        reference: Ground truth reference text.

    Returns:
        1.0 if response contains reference (case-insensitive), 0.0 otherwise.
    """
    if not response or not reference:
        return 0.0

    return 1.0 if reference.strip().lower() in response.lower() else 0.0


def compute_response_length(response: Optional[str]) -> int:
    """
    Compute response length in characters.

    Args:
        response: Generated response text.

    Returns:
        Character count of response.
    """
    return len(response) if response else 0


def compute_token_count(response: Optional[str]) -> int:
    """
    Compute approximate token count (whitespace-based).

    Args:
        response: Generated response text.

    Returns:
        Approximate token count.
    """
    if not response:
        return 0
    return len(response.split())


class DeterministicMetricsCalculator:
    """Calculates deterministic metrics for model responses."""

    def __init__(self, catalog_name: str = "verdict_dev", spark: Optional[SparkSession] = None):
        """
        Initialize metrics calculator.

        Args:
            catalog_name: Unity Catalog name.
            spark: SparkSession instance.
        """
        self.catalog_name = catalog_name
        self.spark = spark or SparkSession.builder.getOrCreate()

    def calculate_metrics(
        self,
        responses_df: DataFrame,
        include_rouge: bool = True,
        include_exact_match: bool = True,
        include_length: bool = True
    ) -> DataFrame:
        """
        Calculate deterministic metrics for responses.

        Args:
            responses_df: DataFrame with 'response', 'prompt_id', 'response_id'.
                         Optionally 'ground_truth' for ROUGE and exact match.
            include_rouge: Whether to compute ROUGE-L.
            include_exact_match: Whether to compute exact match.
            include_length: Whether to compute response length.

        Returns:
            DataFrame with additional metric columns.
        """
        # Register UDFs
        rouge_udf = F.udf(compute_rouge_l, DoubleType())
        exact_match_udf = F.udf(compute_exact_match, DoubleType())
        contains_udf = F.udf(compute_contains_match, DoubleType())
        length_udf = F.udf(compute_response_length, DoubleType())
        token_udf = F.udf(compute_token_count, DoubleType())

        result_df = responses_df

        # Add ground truth column if not present
        if "ground_truth" not in result_df.columns:
            result_df = result_df.withColumn("ground_truth", F.lit(None))

        if include_rouge:
            result_df = result_df.withColumn(
                "rouge_l",
                rouge_udf(F.col("response"), F.col("ground_truth"))
            )

        if include_exact_match:
            result_df = result_df.withColumn(
                "exact_match",
                exact_match_udf(F.col("response"), F.col("ground_truth"))
            )
            result_df = result_df.withColumn(
                "contains_match",
                contains_udf(F.col("response"), F.col("ground_truth"))
            )

        if include_length:
            result_df = result_df.withColumn(
                "response_length",
                length_udf(F.col("response"))
            )
            result_df = result_df.withColumn(
                "token_count",
                token_udf(F.col("response"))
            )

        logger.info(f"Calculated deterministic metrics for {result_df.count()} responses")
        return result_df

    def compute_latency_stats(
        self,
        responses_df: DataFrame,
        group_by: list[str] = None
    ) -> DataFrame:
        """
        Compute latency statistics.

        Args:
            responses_df: DataFrame with 'latency_ms' column.
            group_by: Columns to group by (default: model_version).

        Returns:
            DataFrame with latency statistics (p50, p95, mean, std).
        """
        if group_by is None:
            group_by = ["model_version"]

        stats_df = responses_df.groupBy(*group_by).agg(
            F.expr("percentile_approx(latency_ms, 0.5)").alias("latency_p50"),
            F.expr("percentile_approx(latency_ms, 0.95)").alias("latency_p95"),
            F.mean("latency_ms").alias("latency_mean"),
            F.stddev("latency_ms").alias("latency_std"),
            F.count("*").alias("sample_size")
        )

        return stats_df


def main() -> None:
    """CLI for deterministic metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate deterministic metrics")
    parser.add_argument("--responses-table", required=True, help="Model responses table")
    parser.add_argument("--run-id", help="Filter by run ID")
    parser.add_argument("--catalog", default="verdict_dev", help="Catalog name")

    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    calculator = DeterministicMetricsCalculator(catalog_name=args.catalog, spark=spark)

    # Load responses
    responses_df = spark.table(f"{args.catalog}.{args.responses_table}")
    if args.run_id:
        responses_df = responses_df.filter(F.col("run_id") == args.run_id)

    # Join with ground truth
    prompts_df = spark.table(f"{args.catalog}.raw.prompt_datasets")
    responses_df = responses_df.join(
        prompts_df.select("prompt_id", "ground_truth"),
        on="prompt_id",
        how="left"
    )

    # Calculate metrics
    results = calculator.calculate_metrics(responses_df)
    results.show(10)

    # Latency stats
    stats = calculator.compute_latency_stats(results)
    stats.show()


if __name__ == "__main__":
    main()