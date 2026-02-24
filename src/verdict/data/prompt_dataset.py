"""
Prompt dataset management for Verdict.

Handles loading, versioning, and storing prompt/ground-truth datasets
in Unity Catalog.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    MapType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptDatasetManager:
    """Manages prompt datasets for evaluation."""

    TABLE_NAME = "raw.prompt_datasets"

    def __init__(
        self,
        catalog_name: str = "verdict",
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize prompt dataset manager.

        Args:
            catalog_name: Unity Catalog name.
            spark: SparkSession instance (creates one if not provided).
        """
        self.catalog_name = catalog_name
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.table_full_name = f"{catalog_name}.{self.TABLE_NAME}"

    def create_dataset(
        self,
        prompts: list[dict],
        version: str,
        metadata: Optional[dict[str, str]] = None
    ) -> int:
        """
        Create a new prompt dataset version.

        Args:
            prompts: List of dicts with 'prompt' and optional 'ground_truth'.
                     Each can have 'prompt_id' or one will be generated.
            version: Dataset version string (e.g., 'v1', '2024-01-15').
            metadata: Optional metadata to attach to all prompts.

        Returns:
            Number of prompts loaded.
        """
        records = []
        for p in prompts:
            record = {
                "prompt_id": p.get("prompt_id", str(uuid.uuid4())),
                "dataset_version": version,
                "prompt": p["prompt"],
                "ground_truth": p.get("ground_truth"),
                "metadata": metadata or {},
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            records.append(record)

        schema = StructType([
            StructField("prompt_id", StringType(), False),
            StructField("dataset_version", StringType(), False),
            StructField("prompt", StringType(), False),
            StructField("ground_truth", StringType(), True),
            StructField("metadata", MapType(StringType(), StringType()), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True),
        ])

        df = self.spark.createDataFrame(records, schema)

        # Merge into table for idempotency
        self._merge_prompts(df)

        logger.info(f"Created dataset version '{version}' with {len(records)} prompts")
        return len(records)

    def load_dataset(self, version: str) -> DataFrame:
        """
        Load a specific version of the prompt dataset.

        Args:
            version: Dataset version to load.

        Returns:
            DataFrame with prompts for the specified version.
        """
        df = self.spark.table(self.table_full_name)
        return df.filter(F.col("dataset_version") == version)

    def list_versions(self) -> DataFrame:
        """
        List all dataset versions with counts.

        Returns:
            DataFrame with version, count, and timestamps.
        """
        return (
            self.spark.table(self.table_full_name)
            .groupBy("dataset_version")
            .agg(
                F.count("*").alias("prompt_count"),
                F.min("created_at").alias("created_at"),
                F.max("updated_at").alias("updated_at")
            )
            .orderBy(F.col("created_at").desc())
        )

    def get_latest_version(self) -> Optional[str]:
        """
        Get the most recent dataset version.

        Returns:
            Latest version string or None if no versions exist.
        """
        versions = self.list_versions()
        if versions.count() == 0:
            return None
        return versions.first().dataset_version

    def _merge_prompts(self, df: DataFrame) -> None:
        """
        Merge prompts into the table for idempotency.

        Args:
            df: DataFrame with prompt records to merge.
        """
        df.createOrReplaceTempView("new_prompts")

        merge_sql = f"""
            MERGE INTO {self.table_full_name} AS target
            USING new_prompts AS source
            ON target.prompt_id = source.prompt_id
                AND target.dataset_version = source.dataset_version
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """

        self.spark.sql(merge_sql)
        logger.info("Merged prompts into table")

    def load_from_file(
        self,
        file_path: str,
        version: str,
        format: str = "json",
        prompt_column: str = "prompt",
        ground_truth_column: Optional[str] = "ground_truth",
        metadata: Optional[dict[str, str]] = None
    ) -> int:
        """
        Load prompts from a file.

        Args:
            file_path: Path to the file (DBFS, S3, or local).
            version: Dataset version string.
            format: File format ('json', 'csv', 'parquet').
            prompt_column: Column name containing prompts.
            ground_truth_column: Column name containing ground truth.
            metadata: Optional metadata to attach.

        Returns:
            Number of prompts loaded.
        """
        logger.info(f"Loading prompts from {file_path}")

        reader = self.spark.read.format(format)

        if format == "csv":
            reader = reader.option("header", "true")

        source_df = reader.load(file_path)

        # Transform to expected schema
        df = source_df.select(
            F.col(prompt_column).alias("prompt"),
            F.col(ground_truth_column).alias("ground_truth") if ground_truth_column
            else F.lit(None).alias("ground_truth")
        )

        prompts = [
            {"prompt": row.prompt, "ground_truth": row.ground_truth}
            for row in df.collect()
        ]

        return self.create_dataset(prompts, version, metadata)


def main() -> None:
    """CLI for prompt dataset management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage prompt datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new dataset version")
    create_parser.add_argument("--version", required=True, help="Dataset version")
    create_parser.add_argument("--file", required=True, help="Path to prompts file")
    create_parser.add_argument("--format", default="json", help="File format")
    create_parser.add_argument("--prompt-col", default="prompt", help="Prompt column name")
    create_parser.add_argument("--gt-col", default="ground_truth", help="Ground truth column")

    # List command
    subparsers.add_parser("list", help="List all dataset versions")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get prompts for a version")
    get_parser.add_argument("--version", required=True, help="Dataset version")

    args = parser.parse_args()

    catalog = "verdict"
    manager = PromptDatasetManager(catalog_name=catalog)

    if args.command == "create":
        count = manager.load_from_file(
            file_path=args.file,
            version=args.version,
            format=args.format,
            prompt_column=args.prompt_col,
            ground_truth_column=args.gt_col
        )
        print(f"Loaded {count} prompts into version {args.version}")

    elif args.command == "list":
        manager.list_versions().show()

    elif args.command == "get":
        manager.load_dataset(args.version).show()


if __name__ == "__main__":
    main()