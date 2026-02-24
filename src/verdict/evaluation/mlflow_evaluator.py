"""
MLflow LLM Evaluate integration for Verdict.

Wraps MLflow's LLM evaluation capabilities with Unity Catalog integration.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

import mlflow
from mlflow.metrics.genai import EvaluationExample, answer_relevance, faithfulness, toxicity
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowEvaluator:
    """MLflow LLM Evaluate integration for Verdict."""

    TABLE_NAME = "evaluated.eval_results"

    def __init__(
        self,
        catalog_name: str = "verdict",
        experiment_path: str = "/verdict/experiments",
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize MLflow evaluator.

        Args:
            catalog_name: Unity Catalog name.
            experiment_path: MLflow experiment path.
            spark: SparkSession instance.
        """
        self.catalog_name = catalog_name
        self.experiment_path = experiment_path
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.table_full_name = f"{catalog_name}.{self.TABLE_NAME}"

        # Set MLflow experiment
        mlflow.set_experiment(experiment_path)

    def evaluate_responses(
        self,
        responses_df: DataFrame,
        run_id: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        sample_size: Optional[int] = None
    ) -> DataFrame:
        """
        Evaluate responses using MLflow LLM Evaluate.

        Args:
            responses_df: DataFrame with 'response', 'prompt', 'prompt_id'.
            run_id: MLflow run ID to associate results with.
            metrics: List of metrics to compute. Default: all built-in.
            sample_size: Optional sample size limit.

        Returns:
            DataFrame with evaluation results.
        """
        run_id = run_id or str(uuid.uuid4())
        metrics = metrics or ["faithfulness", "answer_relevance", "toxicity"]

        logger.info(f"Starting MLflow evaluation for {len(responses_df.collect())} responses")

        # Sample if requested
        if sample_size:
            responses_df = responses_df.limit(sample_size)

        # Convert to pandas for MLflow
        pdf = responses_df.select(
            "prompt_id", "prompt", "response", "ground_truth"
        ).toPandas()

        # Prepare evaluation data
        eval_df = pdf.rename(columns={
            "prompt": "inputs",
            "response": "predictions",
            "ground_truth": "targets"
        })

        # Build metrics list
        eval_metrics = self._build_metrics(metrics)

        # Run evaluation
        with mlflow.start_run(run_name=f"eval_{run_id[:8]}") as run:
            mlflow.log_params({
                "run_id": run_id,
                "metrics": ",".join(metrics),
                "sample_size": len(pdf)
            })

            try:
                results = mlflow.evaluate(
                    data=eval_df,
                    targets="targets",
                    predictions="predictions",
                    extra_metrics=eval_metrics,
                    evaluators="default"
                )

                # Log metrics summary
                if results.metrics:
                    for metric_name, metric_value in results.metrics.items():
                        mlflow.log_metric(f"avg_{metric_name}", metric_value)

                logger.info(f"MLflow evaluation complete. Run ID: {run.info.run_id}")

            except Exception as e:
                logger.error(f"MLflow evaluation failed: {e}")
                mlflow.log_param("error", str(e))
                raise

        # Process results and write to Unity Catalog
        results_df = self._process_results(pdf, run_id, metrics)
        self._write_results(results_df)

        return results_df

    def _build_metrics(self, metric_names: list[str]) -> list:
        """
        Build MLflow metric objects from names.

        Args:
            metric_names: List of metric names.

        Returns:
            List of MLflow metric objects.
        """
        metrics_map = {
            "faithfulness": faithfulness(),
            "answer_relevance": answer_relevance(),
            "toxicity": toxicity()
        }

        return [metrics_map[name] for name in metric_names if name in metrics_map]

    def _process_results(
        self,
        pdf,
        run_id: str,
        metrics: list[str]
    ) -> DataFrame:
        """
        Process evaluation results into Spark DataFrame.

        Args:
            pdf: Pandas DataFrame with evaluation results.
            run_id: Run ID for this evaluation.
            metrics: List of metrics computed.

        Returns:
            Spark DataFrame with structured results.
        """
        records = []

        for _, row in pdf.iterrows():
            prompt_id = row["prompt_id"]
            response = row.get("predictions", "")

            for metric_name in metrics:
                # Extract metric value from results
                # MLflow stores metrics in specific column names
                metric_col = f"{metric_name}/mean"
                metric_value = row.get(metric_col, None)

                records.append({
                    "eval_id": str(uuid.uuid4()),
                    "response_id": str(uuid.uuid4()),  # Would need actual mapping
                    "prompt_id": prompt_id,
                    "model_version": "unknown",  # Would need from responses
                    "run_id": run_id,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value) if metric_value else None,
                    "metric_details": json.dumps({"response": response[:500]}),
                    "evaluator": "mlflow",
                    "created_at": datetime.now()
                })

        return self.spark.createDataFrame(records)

    def _write_results(self, df: DataFrame) -> None:
        """Write evaluation results to Unity Catalog."""
        df.createOrReplaceTempView("new_eval_results")

        merge_sql = f"""
            MERGE INTO {self.table_full_name} AS target
            USING new_eval_results AS source
            ON target.eval_id = source.eval_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """

        self.spark.sql(merge_sql)
        logger.info(f"Wrote {df.count()} evaluation results to {self.table_full_name}")


def main() -> None:
    """CLI for MLflow evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MLflow LLM evaluation")
    parser.add_argument("--responses-table", required=True, help="Model responses table")
    parser.add_argument("--run-id", help="Filter by run ID")
    parser.add_argument("--metrics", nargs="+", default=["faithfulness", "answer_relevance", "toxicity"])
    parser.add_argument("--sample-size", type=int, help="Sample size limit")
    parser.add_argument("--catalog", default="verdict", help="Catalog name")
    parser.add_argument("--experiment", default="/verdict/experiments", help="MLflow experiment path")

    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    # Load responses
    responses_df = spark.table(f"{args.catalog}.{args.responses_table}")
    if args.run_id:
        responses_df = responses_df.filter(F.col("run_id") == args.run_id)

    # Join with prompts
    prompts_df = spark.table(f"{args.catalog}.raw.prompt_datasets")
    responses_df = responses_df.join(
        prompts_df.select("prompt_id", "prompt", "ground_truth"),
        on="prompt_id",
        how="left"
    )

    evaluator = MLflowEvaluator(
        catalog_name=args.catalog,
        experiment_path=args.experiment,
        spark=spark
    )

    results = evaluator.evaluate_responses(
        responses_df=responses_df,
        metrics=args.metrics,
        sample_size=args.sample_size
    )

    results.show(10)


if __name__ == "__main__":
    main()