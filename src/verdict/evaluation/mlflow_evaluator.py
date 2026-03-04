"""
MLflow GenAI Evaluate integration for Verdict.

Wraps MLflow's GenAI evaluation capabilities with Unity Catalog integration.
Updated for MLflow 3.x+ API.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import new MLflow GenAI scorers
try:
    from mlflow.genai.scorers import (
        RelevanceToQuery,
        RetrievalGroundedness,
        Correctness,
        Safety,
    )
    MLFLOW_GENAI_AVAILABLE = True
except ImportError:
    MLFLOW_GENAI_AVAILABLE = False
    logger.warning("mlflow.genai.scorers not available. Some metrics will be disabled.")

# Mapping of old metric names to new scorer classes
METRIC_TO_SCORER = {
    "answer_relevance": "RelevanceToQuery",
    "faithfulness": "RetrievalGroundedness",
    "relevance": "RelevanceToQuery",
    "correctness": "Correctness",
    "safety": "Safety",
}


class MLflowEvaluator:
    """MLflow GenAI Evaluate integration for Verdict."""

    TABLE_NAME = "evaluated.eval_results"

    def __init__(
        self,
        catalog_name: str = "verdict_dev",
        experiment_path: str = "/verdict/experiments",
        judge_model: str = "databricks-llama-4-maverick",
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize MLflow evaluator.

        Args:
            catalog_name: Unity Catalog name.
            experiment_path: MLflow experiment path.
            judge_model: Model to use for LLM judges (Databricks serving endpoint).
            spark: SparkSession instance.
        """
        self.catalog_name = catalog_name
        self.experiment_path = experiment_path
        self.judge_model = judge_model
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
        Evaluate responses using MLflow GenAI Evaluate.

        Args:
            responses_df: DataFrame with 'response', 'prompt', 'prompt_id'.
            run_id: MLflow run ID to associate results with.
            metrics: List of metrics to compute. Default: all available.
                     Supported: "answer_relevance", "faithfulness", "correctness", "safety"

        Returns:
            DataFrame with evaluation results.
        """
        run_id = run_id or str(uuid.uuid4())

        if not MLFLOW_GENAI_AVAILABLE:
            logger.warning("MLflow GenAI not available. Returning empty results.")
            return self._empty_results()

        # Default metrics
        if metrics is None:
            metrics = ["answer_relevance", "faithfulness"]

        # Map metric names to scorers
        scorers = self._build_scorers(metrics)
        if not scorers:
            logger.warning("No valid scorers available. Returning empty results.")
            return self._empty_results()

        logger.info(f"Starting MLflow GenAI evaluation for metrics: {metrics}")

        # Sample if requested
        if sample_size:
            responses_df = responses_df.limit(sample_size)

        # Convert to evaluation dataset format
        eval_data = self._prepare_eval_data(responses_df)

        # Run evaluation
        with mlflow.start_run(run_name=f"eval_{run_id[:8]}") as run:
            mlflow.log_params({
                "run_id": run_id,
                "metrics": ",".join(metrics),
                "sample_size": len(eval_data),
                "judge_model": self.judge_model,
            })

            try:
                # Use new mlflow.genai.evaluate API
                results = mlflow.genai.evaluate(
                    data=eval_data,
                    scorers=scorers,
                )

                # Log metrics summary
                if results.metrics:
                    for metric_name, metric_value in results.metrics.items():
                        mlflow.log_metric(f"avg_{metric_name}", metric_value)

                logger.info(f"MLflow GenAI evaluation complete. Run ID: {run.info.run_id}")

            except Exception as e:
                logger.error(f"MLflow GenAI evaluation failed: {e}")
                mlflow.log_param("error", str(e))
                raise

        # Process results and write to Unity Catalog
        results_df = self._process_results(eval_data, run_id, metrics)
        self._write_results(results_df)

        return results_df

    def _build_scorers(self, metric_names: list[str]) -> list:
        """
        Build MLflow GenAI scorer objects from metric names.

        Args:
            metric_names: List of metric names.

        Returns:
            List of MLflow scorer objects.
        """
        scorers = []

        for name in metric_names:
            scorer_name = METRIC_TO_SCORER.get(name)
            if not scorer_name:
                logger.warning(f"Unknown metric: {name}")
                continue

            try:
                if scorer_name == "RelevanceToQuery":
                    scorers.append(RelevanceToQuery())
                elif scorer_name == "RetrievalGroundedness":
                    scorers.append(RetrievalGroundedness())
                elif scorer_name == "Correctness":
                    scorers.append(Correctness(model=self.judge_model))
                elif scorer_name == "Safety":
                    scorers.append(Safety())
            except Exception as e:
                logger.warning(f"Failed to create scorer '{name}': {e}")

        return scorers

    def _prepare_eval_data(self, responses_df: DataFrame) -> list[dict]:
        """
        Convert Spark DataFrame to MLflow GenAI evaluation dataset format.

        Args:
            responses_df: DataFrame with prompt, response, ground_truth columns.

        Returns:
            List of dicts in MLflow GenAI format.
        """
        pdf = responses_df.select(
            "prompt_id", "prompt", "response", "ground_truth"
        ).toPandas()

        eval_data = []
        for _, row in pdf.iterrows():
            eval_data.append({
                "inputs": {"question": row["prompt"]},
                "outputs": row["response"] or "",
                "expectations": {
                    "expected_facts": [row["ground_truth"]] if row.get("ground_truth") else [],
                },
                "metadata": {
                    "prompt_id": row["prompt_id"],
                },
            })

        return eval_data

    def _empty_results(self) -> DataFrame:
        """Create empty results DataFrame."""
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

        schema = StructType([
            StructField("eval_id", StringType(), False),
            StructField("response_id", StringType(), False),
            StructField("prompt_id", StringType(), False),
            StructField("model_version", StringType(), True),
            StructField("run_id", StringType(), False),
            StructField("metric_name", StringType(), False),
            StructField("metric_value", DoubleType(), True),
            StructField("metric_details", StringType(), True),
            StructField("evaluator", StringType(), True),
            StructField("created_at", TimestampType(), False),
        ])
        return self.spark.createDataFrame([], schema)

    def _process_results(
        self,
        eval_data: list[dict],
        run_id: str,
        metrics: list[str]
    ) -> DataFrame:
        """
        Process evaluation results into Spark DataFrame.

        Args:
            eval_data: Evaluation data with results.
            run_id: Run ID for this evaluation.
            metrics: List of metrics computed.

        Returns:
            Spark DataFrame with structured results.
        """
        records = []

        for item in eval_data:
            prompt_id = item.get("metadata", {}).get("prompt_id", str(uuid.uuid4()))
            response = item.get("outputs", "")
            expectations = item.get("expectations", {})

            # Extract scores from results
            # MLflow GenAI stores scores per-item in the evaluation results
            scores = item.get("scores", {})

            for metric_name in metrics:
                score_value = scores.get(metric_name, scores.get(f"{metric_name}/mean"))

                records.append({
                    "eval_id": str(uuid.uuid4()),
                    "response_id": str(uuid.uuid4()),
                    "prompt_id": prompt_id,
                    "model_version": "unknown",
                    "run_id": run_id,
                    "metric_name": metric_name,
                    "metric_value": float(score_value) if score_value is not None else None,
                    "metric_details": json.dumps({
                        "response": response[:500] if response else None,
                        "expected_facts": expectations.get("expected_facts", []),
                    }),
                    "evaluator": "mlflow_genai",
                    "created_at": datetime.now(),
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

    parser = argparse.ArgumentParser(description="Run MLflow GenAI evaluation")
    parser.add_argument("--responses-table", required=True, help="Model responses table")
    parser.add_argument("--run-id", help="Filter by run ID")
    parser.add_argument("--metrics", nargs="+", default=["answer_relevance", "faithfulness"],
                        help="Metrics to compute")
    parser.add_argument("--sample-size", type=int, help="Sample size limit")
    parser.add_argument("--catalog", default="verdict_dev", help="Catalog name")
    parser.add_argument("--experiment", default="/verdict/experiments", help="MLflow experiment path")
    parser.add_argument("--judge-model", default="databricks-llama-4-maverick",
                        help="Judge model endpoint")

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
        judge_model=args.judge_model,
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