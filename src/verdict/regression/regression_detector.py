"""
Regression detection for Verdict.

Compares metric distributions between candidate and baseline model versions
using statistical tests to detect quality regressions.
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

import mlflow
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerdictLabel(str, Enum):
    """Verdict outcome labels."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class RegressionDetector:
    """Detects regressions between model versions."""

    TABLE_NAME = "metrics.metric_summary"

    def __init__(
        self,
        catalog_name: str = "verdict",
        threshold_pct: float = 5.0,
        p_value_threshold: float = 0.05,
        experiment_path: str = "/verdict/experiments",
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize regression detector.

        Args:
            catalog_name: Unity Catalog name.
            threshold_pct: Percentage drop threshold for regression.
            p_value_threshold: Statistical significance threshold.
            experiment_path: MLflow experiment path for artifacts.
            spark: SparkSession instance.
        """
        self.catalog_name = catalog_name
        self.threshold_pct = threshold_pct
        self.p_value_threshold = p_value_threshold
        self.experiment_path = experiment_path
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.table_full_name = f"{catalog_name}.{self.TABLE_NAME}"

        mlflow.set_experiment(experiment_path)

    def detect_regression(
        self,
        candidate_version: str,
        baseline_version: str,
        run_id: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        dataset_version: Optional[str] = None
    ) -> dict:
        """
        Detect regressions between candidate and baseline model versions.

        Args:
            candidate_version: Version string of the candidate model.
            baseline_version: Version string of the baseline model.
            run_id: Optional run ID to filter results.
            metrics: List of metrics to compare (default: all).
            dataset_version: Dataset version to filter by.

        Returns:
            Dict with verdict and detailed regression report.
        """
        run_id = run_id or str(uuid.uuid4())
        metrics = metrics or self._get_default_metrics()

        logger.info(
            f"Detecting regression: candidate={candidate_version}, "
            f"baseline={baseline_version}"
        )

        # Load metric data for both versions
        candidate_data = self._load_metrics(candidate_version, run_id)
        baseline_data = self._load_metrics(baseline_version, run_id)

        # Compare each metric
        comparisons = []
        for metric_name in metrics:
            comparison = self._compare_metric(
                metric_name=metric_name,
                candidate_values=candidate_data.get(metric_name, []),
                baseline_values=baseline_data.get(metric_name, []),
                candidate_version=candidate_version,
                baseline_version=baseline_version
            )
            comparisons.append(comparison)

        # Determine overall verdict
        verdict = self._determine_verdict(comparisons)

        report = {
            "summary_id": str(uuid.uuid4()),
            "run_id": run_id,
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "dataset_version": dataset_version,
            "verdict": verdict.value,
            "comparisons": comparisons,
            "timestamp": datetime.now().isoformat()
        }

        # Log to MLflow and write to Unity Catalog
        self._log_to_mlflow(report)
        self._write_summary(report)

        logger.info(f"Regression detection complete: {verdict.value}")
        return report

    def _load_metrics(self, model_version: str, run_id: Optional[str]) -> dict[str, list]:
        """
        Load metric values for a model version.

        Args:
            model_version: Model version to load metrics for.
            run_id: Optional run ID filter.

        Returns:
            Dict mapping metric names to lists of values.
        """
        eval_table = f"{self.catalog_name}.evaluated.eval_results"

        df = self.spark.table(eval_table).filter(
            F.col("model_version") == model_version
        )

        if run_id:
            df = df.filter(F.col("run_id") == run_id)

        # Aggregate metrics
        metrics_data = {}
        for row in df.collect():
            metric_name = row.metric_name
            if metric_name not in metrics_data:
                metrics_data[metric_name] = []
            if row.metric_value is not None:
                metrics_data[metric_name].append(row.metric_value)

        return metrics_data

    def _compare_metric(
        self,
        metric_name: str,
        candidate_values: list,
        baseline_values: list,
        candidate_version: str,
        baseline_version: str
    ) -> dict:
        """
        Compare a single metric between candidate and baseline.

        Uses Mann-Whitney U test for statistical comparison.

        Args:
            metric_name: Name of the metric.
            candidate_values: Values from candidate model.
            baseline_values: Values from baseline model.
            candidate_version: Candidate version string.
            baseline_version: Baseline version string.

        Returns:
            Dict with comparison results.
        """
        if not candidate_values or not baseline_values:
            return {
                "metric_name": metric_name,
                "candidate_mean": None,
                "baseline_mean": None,
                "pct_change": None,
                "p_value": None,
                "is_regression": None,
                "reason": "Insufficient data"
            }

        candidate_mean = np.mean(candidate_values)
        baseline_mean = np.mean(baseline_values)

        # Calculate percentage change (negative = improvement for most metrics)
        if baseline_mean != 0:
            pct_change = ((candidate_mean - baseline_mean) / baseline_mean) * 100
        else:
            pct_change = 0.0

        # Mann-Whitney U test
        if len(candidate_values) >= 2 and len(baseline_values) >= 2:
            try:
                stat, p_value = stats.mannwhitneyu(
                    candidate_values,
                    baseline_values,
                    alternative='two-sided'
                )
            except Exception:
                p_value = 1.0
        else:
            p_value = 1.0

        # Determine if regression
        # For most metrics (higher = better), a significant drop is regression
        # For latency (lower = better), a significant increase is regression
        is_latency = "latency" in metric_name.lower()

        if is_latency:
            # Higher latency is worse
            is_regression = (
                pct_change > self.threshold_pct and
                p_value < self.p_value_threshold
            )
        else:
            # Lower metric score is worse (for quality metrics)
            is_regression = (
                pct_change < -self.threshold_pct and
                p_value < self.p_value_threshold
            )

        return {
            "metric_name": metric_name,
            "candidate_mean": float(candidate_mean),
            "baseline_mean": float(baseline_mean),
            "candidate_std": float(np.std(candidate_values)),
            "baseline_std": float(np.std(baseline_values)),
            "pct_change": float(pct_change),
            "p_value": float(p_value),
            "is_regression": is_regression,
            "sample_size_candidate": len(candidate_values),
            "sample_size_baseline": len(baseline_values)
        }

    def _determine_verdict(self, comparisons: list[dict]) -> VerdictLabel:
        """
        Determine overall verdict from metric comparisons.

        Args:
            comparisons: List of metric comparison results.

        Returns:
            Overall verdict label.
        """
        regressions = [c for c in comparisons if c.get("is_regression")]

        if not regressions:
            return VerdictLabel.PASS

        # Check severity of regressions
        severe_regressions = [
            r for r in regressions
            if abs(r.get("pct_change", 0)) > self.threshold_pct * 2
        ]

        if severe_regressions:
            return VerdictLabel.FAIL

        return VerdictLabel.WARN

    def _log_to_mlflow(self, report: dict) -> None:
        """Log regression report to MLflow."""
        with mlflow.start_run(run_name=f"regression_{report['run_id'][:8]}") as run:
            mlflow.log_params({
                "candidate_version": report["candidate_version"],
                "baseline_version": report["baseline_version"],
                "verdict": report["verdict"]
            })

            # Log metrics
            for comp in report["comparisons"]:
                metric_name = comp["metric_name"]
                mlflow.log_metric(f"{metric_name}_candidate_mean", comp.get("candidate_mean", 0))
                mlflow.log_metric(f"{metric_name}_baseline_mean", comp.get("baseline_mean", 0))
                mlflow.log_metric(f"{metric_name}_pct_change", comp.get("pct_change", 0))

            # Log report as artifact
            report_json = json.dumps(report, indent=2)
            mlflow.log_text(report_json, "regression_report.json")

            logger.info(f"Logged regression report to MLflow run: {run.info.run_id}")

    def _write_summary(self, report: dict) -> None:
        """Write regression summary to Unity Catalog."""
        records = []

        for comp in report["comparisons"]:
            records.append({
                "summary_id": report["summary_id"],
                "run_id": report["run_id"],
                "model_version": report["candidate_version"],
                "dataset_version": report.get("dataset_version", "unknown"),
                "metric_name": comp["metric_name"],
                "metric_mean": comp.get("candidate_mean"),
                "metric_std": comp.get("candidate_std"),
                "metric_p50": comp.get("candidate_mean"),  # Approximate
                "metric_p95": None,  # Would need raw data
                "sample_size": comp.get("sample_size_candidate"),
                "verdict": report["verdict"],
                "verdict_reason": json.dumps({
                    "pct_change": comp.get("pct_change"),
                    "p_value": comp.get("p_value"),
                    "is_regression": comp.get("is_regression")
                }),
                "baseline_version": report["baseline_version"],
                "baseline_mean": comp.get("baseline_mean"),
                "p_value": comp.get("p_value"),
                "created_at": datetime.now()
            })

        df = self.spark.createDataFrame(records)
        df.createOrReplaceTempView("new_summary")

        merge_sql = f"""
            MERGE INTO {self.table_full_name} AS target
            USING new_summary AS source
            ON target.summary_id = source.summary_id
                AND target.metric_name = source.metric_name
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """

        self.spark.sql(merge_sql)
        logger.info(f"Wrote regression summary to {self.table_full_name}")

    def _get_default_metrics(self) -> list[str]:
        """Get default metrics to compare."""
        return [
            "faithfulness",
            "answer_relevance",
            "judge_score",
            "rouge_l",
            "latency_p95"
        ]

    def get_verdict_history(
        self,
        model_version: Optional[str] = None,
        limit: int = 100
    ) -> DataFrame:
        """
        Get verdict history for a model.

        Args:
            model_version: Optional model version filter.
            limit: Maximum number of records.

        Returns:
            DataFrame with verdict history.
        """
        df = self.spark.table(self.table_full_name)

        if model_version:
            df = df.filter(F.col("model_version") == model_version)

        return df.orderBy(F.col("created_at").desc()).limit(limit)


def main() -> None:
    """CLI for regression detector."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect regressions between model versions")
    parser.add_argument("--candidate", required=True, help="Candidate model version")
    parser.add_argument("--baseline", required=True, help="Baseline model version")
    parser.add_argument("--run-id", help="Evaluation run ID")
    parser.add_argument("--threshold", type=float, default=5.0, help="Regression threshold %%")
    parser.add_argument("--p-value", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("--catalog", default="verdict", help="Catalog name")

    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    detector = RegressionDetector(
        catalog_name=args.catalog,
        threshold_pct=args.threshold,
        p_value_threshold=args.p_value,
        spark=spark
    )

    report = detector.detect_regression(
        candidate_version=args.candidate,
        baseline_version=args.baseline,
        run_id=args.run_id
    )

    print(f"\n{'='*60}")
    print(f"VERDICT: {report['verdict']}")
    print(f"{'='*60}")

    for comp in report["comparisons"]:
        status = "⚠️ REGRESSION" if comp.get("is_regression") else "✓ OK"
        print(f"\n{comp['metric_name']}: {status}")
        print(f"  Candidate: {comp.get('candidate_mean', 'N/A'):.3f}")
        print(f"  Baseline:  {comp.get('baseline_mean', 'N/A'):.3f}")
        print(f"  Change:    {comp.get('pct_change', 0):.1f}%")
        print(f"  P-value:   {comp.get('p_value', 1):.4f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()