"""
Inference runner for Verdict.

Runs parallel inference against Databricks Model Serving endpoints
using Spark UDFs for scalability.
"""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRunner:
    """Runs inference against Databricks Model Serving endpoints."""

    TABLE_NAME = "raw.model_responses"

    def __init__(
        self,
        catalog_name: str = "verdict",
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize inference runner.

        Args:
            catalog_name: Unity Catalog name.
            spark: SparkSession instance.
        """
        self.catalog_name = catalog_name
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.table_full_name = f"{catalog_name}.{self.TABLE_NAME}"

    def run_inference(
        self,
        endpoint_name: str,
        prompt_dataset: DataFrame,
        model_version: Optional[str] = None,
        run_id: Optional[str] = None,
        batch_size: int = 100,
        max_workers: int = 10,
        api_token_secret: Optional[str] = None,
        databricks_host: Optional[str] = None
    ) -> DataFrame:
        """
        Run inference on prompts against a model endpoint.

        Args:
            endpoint_name: Name of the Model Serving endpoint.
            prompt_dataset: DataFrame with 'prompt_id' and 'prompt' columns.
            model_version: Version string for the model (auto-detected if not provided).
            run_id: MLflow run ID for this inference run.
            batch_size: Number of prompts to process per batch.
            max_workers: Maximum parallel workers for inference.
            api_token_secret: Databricks secret scope/key for API token.
            databricks_host: Databricks workspace URL (auto-detected if not provided).

        Returns:
            DataFrame with inference results.
        """
        run_id = run_id or str(uuid.uuid4())

        # Get API token and host
        api_token = self._get_api_token(api_token_secret)
        host = databricks_host or self._get_databricks_host()

        # Get model version from endpoint if not provided
        if not model_version:
            model_version = self._get_endpoint_version(endpoint_name, api_token, host)

        logger.info(
            f"Starting inference on endpoint '{endpoint_name}' "
            f"(version: {model_version}, run_id: {run_id})"
        )

        # Collect prompts for parallel processing
        prompts = prompt_dataset.select("prompt_id", "prompt").collect()
        total_prompts = len(prompts)
        logger.info(f"Processing {total_prompts} prompts")

        # Run inference in parallel
        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._call_endpoint,
                    endpoint_name=endpoint_name,
                    prompt_id=row.prompt_id,
                    prompt=row.prompt,
                    api_token=api_token,
                    host=host
                ): row.prompt_id
                for row in prompts
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    prompt_id = futures[future]
                    logger.error(f"Inference failed for prompt {prompt_id}: {e}")
                    results.append({
                        "response_id": str(uuid.uuid4()),
                        "prompt_id": prompt_id,
                        "response": None,
                        "latency_ms": None,
                        "status": "error",
                        "error_message": str(e)
                    })

                if i % batch_size == 0:
                    logger.info(f"Processed {i}/{total_prompts} prompts")

        elapsed = time.time() - start_time
        logger.info(f"Inference complete in {elapsed:.2f}s ({total_prompts/elapsed:.1f} prompts/sec)")

        # Create results DataFrame
        results_df = self._create_results_dataframe(
            results=results,
            model_version=model_version,
            endpoint_name=endpoint_name,
            run_id=run_id
        )

        # Write to Delta table
        self._write_results(results_df)

        return results_df

    def _call_endpoint(
        self,
        endpoint_name: str,
        prompt_id: str,
        prompt: str,
        api_token: str,
        host: str
    ) -> dict:
        """
        Call the model serving endpoint for a single prompt.

        Args:
            endpoint_name: Model Serving endpoint name.
            prompt_id: Unique prompt identifier.
            prompt: The prompt text.
            api_token: Databricks API token.
            host: Databricks workspace URL.

        Returns:
            Dict with response details.
        """
        url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [{"role": "user", "content": prompt}]
        }

        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                # Extract text from response (format varies by model)
                output_text = self._extract_response_text(result)
                return {
                    "response_id": str(uuid.uuid4()),
                    "prompt_id": prompt_id,
                    "response": output_text,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "error_message": None
                }
            else:
                return {
                    "response_id": str(uuid.uuid4()),
                    "prompt_id": prompt_id,
                    "response": None,
                    "latency_ms": latency_ms,
                    "status": f"http_{response.status_code}",
                    "error_message": response.text[:500]
                }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "response_id": str(uuid.uuid4()),
                "prompt_id": prompt_id,
                "response": None,
                "latency_ms": latency_ms,
                "status": "error",
                "error_message": str(e)
            }

    def _extract_response_text(self, response: dict) -> str:
        """
        Extract text from model response.

        Handles different response formats from various model types.

        Args:
            response: Raw response dict from endpoint.

        Returns:
            Extracted text string.
        """
        # Try common response formats
        if "predictions" in response:
            predictions = response["predictions"]
            if isinstance(predictions, list) and len(predictions) > 0:
                pred = predictions[0]
                if isinstance(pred, str):
                    return pred
                if isinstance(pred, dict) and "content" in pred:
                    return pred["content"]

        if "choices" in response:
            choices = response["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                if "text" in choice:
                    return choice["text"]

        # Fallback: stringify the response
        return json.dumps(response)

    def _create_results_dataframe(
        self,
        results: list[dict],
        model_version: str,
        endpoint_name: str,
        run_id: str
    ) -> DataFrame:
        """Create a Spark DataFrame from inference results."""
        records = []
        for r in results:
            records.append({
                "response_id": r["response_id"],
                "prompt_id": r["prompt_id"],
                "model_version": model_version,
                "endpoint_name": endpoint_name,
                "response": r["response"],
                "latency_ms": r["latency_ms"],
                "status": r["status"],
                "error_message": r["error_message"],
                "created_at": datetime.now(),
                "run_id": run_id
            })

        schema = StructType([
            StructField("response_id", StringType(), False),
            StructField("prompt_id", StringType(), False),
            StructField("model_version", StringType(), False),
            StructField("endpoint_name", StringType(), False),
            StructField("response", StringType(), True),
            StructField("latency_ms", DoubleType(), True),
            StructField("status", StringType(), True),
            StructField("error_message", StringType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("run_id", StringType(), True),
        ])

        return self.spark.createDataFrame(records, schema)

    def _write_results(self, df: DataFrame) -> None:
        """Write inference results to Delta table using MERGE."""
        df.createOrReplaceTempView("new_responses")

        merge_sql = f"""
            MERGE INTO {self.table_full_name} AS target
            USING new_responses AS source
            ON target.response_id = source.response_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """

        self.spark.sql(merge_sql)
        logger.info(f"Wrote {df.count()} results to {self.table_full_name}")

    def _get_api_token(self, secret_scope_key: Optional[str]) -> str:
        """Get API token from Databricks secrets."""
        if secret_scope_key:
            try:
                from dbutils import DBUtils  # type: ignore
                dbutils = DBUtils()
                scope, key = secret_scope_key.split("/")
                return dbutils.secrets.get(scope, key)
            except Exception:
                pass

        # Try environment variable
        import os
        token = os.environ.get("DATABRICKS_TOKEN")
        if token:
            return token

        raise ValueError(
            "API token not found. Set DATABRICKS_TOKEN env var or provide secret scope/key."
        )

    def _get_databricks_host(self) -> str:
        """Get Databricks workspace URL."""
        import os

        host = os.environ.get("DATABRICKS_HOST")
        if host:
            return host.rstrip("/")

        # Try Spark config
        spark_conf = self.spark.conf.get("spark.databricks.workspaceUrl", None)
        if spark_conf:
            return f"https://{spark_conf}"

        raise ValueError(
            "Databricks host not found. Set DATABRICKS_HOST env var."
        )

    def _get_endpoint_version(self, endpoint_name: str, api_token: str, host: str) -> str:
        """Get the current model version from endpoint."""
        url = f"{host}/api/2.0/serving-endpoints/{endpoint_name}"
        headers = {"Authorization": f"Bearer {api_token}"}

        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                # Extract version from endpoint config
                if "pending_config" in data:
                    return data["pending_config"].get("served_models", [{}])[0].get("version", "unknown")
                if "config" in data:
                    return data["config"].get("served_models", [{}])[0].get("version", "unknown")
        except Exception as e:
            logger.warning(f"Could not fetch endpoint version: {e}")

        return "unknown"


def main() -> None:
    """CLI for inference runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on model endpoints")
    parser.add_argument("--endpoint", required=True, help="Model serving endpoint name")
    parser.add_argument("--dataset-version", required=True, help="Prompt dataset version")
    parser.add_argument("--model-version", help="Model version (auto-detected if not provided)")
    parser.add_argument("--run-id", help="MLflow run ID")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--catalog", default="verdict", help="Catalog name")

    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    runner = InferenceRunner(catalog_name=args.catalog, spark=spark)

    # Load prompt dataset
    from data.prompt_dataset import PromptDatasetManager
    dataset_manager = PromptDatasetManager(catalog_name=args.catalog, spark=spark)
    prompts = dataset_manager.load_dataset(args.dataset_version)

    # Run inference
    results = runner.run_inference(
        endpoint_name=args.endpoint,
        prompt_dataset=prompts,
        model_version=args.model_version,
        run_id=args.run_id,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )

    results.show(10)


if __name__ == "__main__":
    main()