"""
Custom LLM-as-a-judge scorers for Verdict.

Implements LLM-based evaluation where another LLM judges the quality
of responses.
"""

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

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

# Default judge prompt template
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator judging the quality of an AI assistant's response.

**Prompt:** {prompt}

**Response:** {response}

**Ground Truth (if available):** {ground_truth}

Evaluate the response on a scale of 1-5:
- 1: Poor - Completely irrelevant, incorrect, or unhelpful
- 2: Below Average - Major issues with accuracy or relevance
- 3: Average - Adequate response with some issues
- 4: Good - Accurate and helpful with minor issues
- 5: Excellent - Perfect or near-perfect response

Provide your evaluation in JSON format:
{{
  "score": <1-5>,
  "reasoning": "<brief explanation>",
  "strengths": ["<strength1>", "<strength2>"],
  "weaknesses": ["<weakness1>"]
}}

Respond with ONLY the JSON object, no additional text."""


class LLMJudge:
    """LLM-as-a-judge scorer."""

    def __init__(
        self,
        endpoint_name: str = "databricks-llama-4-maverick",
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_workers: int = 10
    ):
        """
        Initialize LLM judge.

        Args:
            endpoint_name: Model Serving endpoint for the judge.
            api_token: Databricks API token.
            host: Databricks workspace URL.
            prompt_template: Custom prompt template for judging.
            max_workers: Maximum parallel workers.
        """
        self.endpoint_name = endpoint_name
        self.api_token = api_token or self._get_api_token()
        self.host = host or self._get_host()
        self.prompt_template = prompt_template or JUDGE_PROMPT_TEMPLATE
        self.max_workers = max_workers

    def judge(
        self,
        prompt: str,
        response: str,
        ground_truth: Optional[str] = None
    ) -> dict:
        """
        Judge a single prompt/response pair.

        Args:
            prompt: The original prompt.
            response: The model's response.
            ground_truth: Optional ground truth reference.

        Returns:
            Dict with 'score', 'reasoning', 'strengths', 'weaknesses'.
        """
        judge_prompt = self.prompt_template.format(
            prompt=prompt,
            response=response,
            ground_truth=ground_truth or "N/A"
        )

        try:
            result = self._call_endpoint(judge_prompt)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "score": None,
                "reasoning": f"Evaluation failed: {str(e)}",
                "strengths": [],
                "weaknesses": [],
                "error": str(e)
            }

    def judge_batch(
        self,
        items: list[dict]
    ) -> list[dict]:
        """
        Judge multiple prompt/response pairs in parallel.

        Args:
            items: List of dicts with 'prompt', 'response', optional 'ground_truth'.

        Returns:
            List of evaluation results.
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.judge,
                    prompt=item["prompt"],
                    response=item["response"],
                    ground_truth=item.get("ground_truth")
                ): i
                for i, item in enumerate(items)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Batch judge failed for item {idx}: {e}")
                    results.append((idx, {
                        "score": None,
                        "reasoning": f"Evaluation failed: {str(e)}",
                        "error": str(e)
                    }))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _call_endpoint(self, prompt: str) -> str:
        """Call the judge model endpoint."""
        url = f"{self.host}/serving-endpoints/{self.endpoint_name}/invocations"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": [{"role": "user", "content": prompt}]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"Endpoint returned {response.status_code}: {response.text}")

        result = response.json()
        return self._extract_text(result)

    def _extract_text(self, response: dict) -> str:
        """Extract text from model response."""
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

        return json.dumps(response)

    def _parse_result(self, result: str) -> dict:
        """Parse judge result from model output."""
        # Try to extract JSON from response
        try:
            # Find JSON object in response
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                parsed = json.loads(json_str)
                return {
                    "score": float(parsed.get("score", 0)),
                    "reasoning": parsed.get("reasoning", ""),
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", [])
                }
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract score from text
        import re
        score_match = re.search(r'"score"\s*:\s*(\d)', result)
        if score_match:
            return {
                "score": float(score_match.group(1)),
                "reasoning": result,
                "strengths": [],
                "weaknesses": []
            }

        return {
            "score": None,
            "reasoning": f"Could not parse judge response: {result[:200]}",
            "strengths": [],
            "weaknesses": []
        }

    def _get_api_token(self) -> str:
        """Get API token from environment."""
        import os
        token = os.environ.get("DATABRICKS_TOKEN")
        if not token:
            raise ValueError("DATABRICKS_TOKEN not set")
        return token

    def _get_host(self) -> str:
        """Get Databricks host from environment."""
        import os
        host = os.environ.get("DATABRICKS_HOST")
        if not host:
            raise ValueError("DATABRICKS_HOST not set")
        return host.rstrip("/")


class LLMJudgeEvaluator:
    """Evaluates responses using LLM-as-a-judge."""

    TABLE_NAME = "evaluated.eval_results"

    def __init__(
        self,
        catalog_name: str = "verdict",
        judge_endpoint: str = "databricks-llama-4-maverick",
        spark: Optional[SparkSession] = None,
        max_workers: int = 10
    ):
        """
        Initialize LLM judge evaluator.

        Args:
            catalog_name: Unity Catalog name.
            judge_endpoint: Judge model endpoint.
            spark: SparkSession instance.
            max_workers: Maximum parallel workers.
        """
        self.catalog_name = catalog_name
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.table_full_name = f"{catalog_name}.{self.TABLE_NAME}"
        self.judge = LLMJudge(endpoint_name=judge_endpoint, max_workers=max_workers)

    def evaluate(
        self,
        responses_df: DataFrame,
        run_id: Optional[str] = None,
        sample_size: Optional[int] = None
    ) -> DataFrame:
        """
        Evaluate responses using LLM-as-a-judge.

        Args:
            responses_df: DataFrame with 'prompt_id', 'prompt', 'response', 'ground_truth'.
            run_id: Run ID for this evaluation.
            sample_size: Optional sample size limit.

        Returns:
            DataFrame with judge evaluation results.
        """
        run_id = run_id or str(uuid.uuid4())

        # Sample if requested
        if sample_size:
            responses_df = responses_df.limit(sample_size)

        # Collect for batch processing
        rows = responses_df.select(
            "prompt_id", "prompt", "response", "ground_truth", "model_version"
        ).collect()

        logger.info(f"Evaluating {len(rows)} responses with LLM judge")

        # Prepare items for batch judging
        items = [
            {
                "prompt": row.prompt,
                "response": row.response,
                "ground_truth": row.ground_truth
            }
            for row in rows
        ]

        # Run batch evaluation
        results = self.judge.judge_batch(items)

        # Create results DataFrame
        records = []
        for row, result in zip(rows, results):
            records.append({
                "eval_id": str(uuid.uuid4()),
                "response_id": str(uuid.uuid4()),  # Would need actual mapping
                "prompt_id": row.prompt_id,
                "model_version": row.model_version,
                "run_id": run_id,
                "metric_name": "judge_score",
                "metric_value": result["score"],
                "metric_details": json.dumps({
                    "reasoning": result.get("reasoning"),
                    "strengths": result.get("strengths", []),
                    "weaknesses": result.get("weaknesses", [])
                }),
                "evaluator": f"llm_judge:{self.judge.endpoint_name}",
                "created_at": datetime.now()
            })

        results_df = self.spark.createDataFrame(records)
        self._write_results(results_df)

        logger.info(f"Wrote {len(records)} judge evaluation results")
        return results_df

    def _write_results(self, df: DataFrame) -> None:
        """Write evaluation results to Unity Catalog."""
        df.createOrReplaceTempView("new_judge_results")

        merge_sql = f"""
            MERGE INTO {self.table_full_name} AS target
            USING new_judge_results AS source
            ON target.eval_id = source.eval_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """

        self.spark.sql(merge_sql)


def main() -> None:
    """CLI for LLM judge evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation")
    parser.add_argument("--responses-table", required=True, help="Model responses table")
    parser.add_argument("--run-id", help="Filter by run ID")
    parser.add_argument("--judge-endpoint", default="databricks-llama-4-maverick", help="Judge model endpoint")
    parser.add_argument("--sample-size", type=int, help="Sample size limit")
    parser.add_argument("--catalog", default="verdict", help="Catalog name")

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

    evaluator = LLMJudgeEvaluator(
        catalog_name=args.catalog,
        judge_endpoint=args.judge_endpoint,
        spark=spark
    )

    results = evaluator.evaluate(
        responses_df=responses_df,
        sample_size=args.sample_size
    )

    results.show(10)


if __name__ == "__main__":
    main()