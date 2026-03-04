"""
Testgen chunks metadata management for Verdict.

Handles storing and retrieving source chunks used for test dataset generation.
"""

import logging
from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestgenChunksManager:
    """Manages source chunks for test dataset generation."""

    TABLE_NAME = "metadata.testgen_chunks"

    def __init__(
        self,
        catalog_name: str = "verdict_dev",
        spark: Optional[SparkSession] = None
    ):
        """
        Initialize testgen chunks manager.

        Args:
            catalog_name: Unity Catalog name.
            spark: SparkSession instance (creates one if not provided).
        """
        self.catalog_name = catalog_name
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.table_full_name = f"{catalog_name}.{self.TABLE_NAME}"

    def save_chunks(
        self,
        chunks: list[dict],
        dataset_version: str,
        source_collection: str,
        qa_pairs_count: dict[str, int] | None = None,
    ) -> int:
        """
        Save source chunks to metadata table.

        Args:
            chunks: List of chunk dicts with 'id', 'text', 'metadata'.
            dataset_version: Dataset version string.
            source_collection: Qdrant collection name.
            qa_pairs_count: Optional dict mapping chunk_id to Q&A pair count.

        Returns:
            Number of chunks saved.
        """
        qa_pairs_count = qa_pairs_count or {}
        records = []

        for chunk in chunks:
            # Count Q&A pairs for this chunk
            qa_count = qa_pairs_count.get(chunk["id"], 0)

            record = {
                "chunk_id": chunk["id"],
                "dataset_version": dataset_version,
                "source_collection": source_collection,
                "source_text": chunk.get("text", ""),
                "chunk_metadata": chunk.get("metadata", {}),
                "qa_pairs_count": qa_count,
                "created_at": datetime.now(),
            }
            records.append(record)

        schema = StructType([
            StructField("chunk_id", StringType(), False),
            StructField("dataset_version", StringType(), False),
            StructField("source_collection", StringType(), False),
            StructField("source_text", StringType(), True),
            StructField("chunk_metadata", MapType(StringType(), StringType()), True),
            StructField("qa_pairs_count", IntegerType(), True),
            StructField("created_at", TimestampType(), False),
        ])

        df = self.spark.createDataFrame(records, schema)
        self._merge_chunks(df)

        logger.info(f"Saved {len(records)} chunks for version '{dataset_version}'")
        return len(records)

    def load_chunks(self, dataset_version: str) -> DataFrame:
        """
        Load chunks for a specific dataset version.

        Args:
            dataset_version: Dataset version to load.

        Returns:
            DataFrame with chunks for the specified version.
        """
        df = self.spark.table(self.table_full_name)
        return df.filter(F.col("dataset_version") == dataset_version)

    def list_versions(self) -> DataFrame:
        """
        List all dataset versions with chunk counts.

        Returns:
            DataFrame with version, chunk_count, total_qa_pairs, and timestamps.
        """
        return (
            self.spark.table(self.table_full_name)
            .groupBy("dataset_version", "source_collection")
            .agg(
                F.count("*").alias("chunk_count"),
                F.sum("qa_pairs_count").alias("total_qa_pairs"),
                F.min("created_at").alias("created_at"),
                F.max("created_at").alias("updated_at")
            )
            .orderBy(F.col("created_at").desc())
        )

    def get_chunk_texts(self, dataset_version: str) -> list[dict]:
        """
        Get chunk texts for a dataset version.

        Args:
            dataset_version: Dataset version to load.

        Returns:
            List of dicts with chunk_id and source_text.
        """
        df = self.load_chunks(dataset_version)
        rows = df.select("chunk_id", "source_text").collect()
        return [
            {"chunk_id": row.chunk_id, "source_text": row.source_text}
            for row in rows
        ]

    def get_generation_stats(self, dataset_version: str) -> dict:
        """
        Get generation statistics for a dataset version.

        Args:
            dataset_version: Dataset version.

        Returns:
            Dict with generation statistics.
        """
        df = self.load_chunks(dataset_version)

        stats = df.agg(
            F.count("*").alias("total_chunks"),
            F.sum("qa_pairs_count").alias("total_qa_pairs"),
            F.avg("qa_pairs_count").alias("avg_qa_per_chunk"),
        ).first()

        return {
            "dataset_version": dataset_version,
            "total_chunks": stats.total_chunks or 0,
            "total_qa_pairs": stats.total_qa_pairs or 0,
            "avg_qa_per_chunk": float(stats.avg_qa_per_chunk) if stats.avg_qa_per_chunk else 0.0,
        }

    def _merge_chunks(self, df: DataFrame) -> None:
        """
        Merge chunks into the table for idempotency.

        Args:
            df: DataFrame with chunk records to merge.
        """
        df.createOrReplaceTempView("new_chunks")

        merge_sql = f"""
            MERGE INTO {self.table_full_name} AS target
            USING new_chunks AS source
            ON target.chunk_id = source.chunk_id
                AND target.dataset_version = source.dataset_version
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """

        self.spark.sql(merge_sql)
        logger.info("Merged chunks into table")