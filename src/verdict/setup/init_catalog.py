"""
Unity Catalog initialization for Verdict.

Creates the catalog, schemas, and Delta tables required for the
LLMOps evaluation framework using PySpark directly.

Can be run as a Databricks notebook or Python module.
"""

import logging

from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerdictCatalogSetup:
    """Manages Unity Catalog setup for Verdict using PySpark."""

    def __init__(self, catalog_name: str = "verdict", spark: SparkSession | None = None):
        """
        Initialize catalog setup.

        Args:
            catalog_name: Name of the Unity Catalog catalog.
            spark: SparkSession instance (creates one if not provided).
        """
        self.catalog_name = catalog_name
        self.spark = spark or SparkSession.builder.getOrCreate()
        self.schemas = ["raw", "evaluated", "metrics"]

    def catalog_exists(self) -> bool:
        """Check if the catalog already exists."""
        try:
            catalogs = self.spark.sql("SHOW CATALOGS").collect()
            return any(row.catalog == self.catalog_name for row in catalogs)
        except Exception:
            return False

    def create_catalog(self) -> None:
        """Create the verdict catalog if it doesn't exist."""
        # Check if catalog exists first to avoid storage location errors
        if self.catalog_exists():
            logger.info(f"Catalog '{self.catalog_name}' already exists, skipping creation")
            return

        try:
            self.spark.sql(f"CREATE CATALOG {self.catalog_name}")
            logger.info(f"Created catalog '{self.catalog_name}'")
        except Exception as e:
            logger.warning(
                f"Could not create catalog '{self.catalog_name}': {e}\n"
                "Please create the catalog manually via Databricks UI if needed.\n"
                "Continuing with schema and table setup..."
            )

    def create_schemas(self) -> None:
        """Create schemas within the catalog."""
        for schema_name in self.schemas:
            full_name = f"{self.catalog_name}.{schema_name}"
            try:
                self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {full_name}")
                logger.info(f"Schema '{full_name}' created or already exists")
            except Exception as e:
                logger.error(f"Failed to create schema '{full_name}': {e}")
                raise

    def create_tables(self) -> None:
        """Create Delta tables for Verdict."""
        tables = self._get_table_definitions()

        for table_name, ddl in tables.items():
            full_name = f"{self.catalog_name}.{table_name}"
            logger.info(f"Creating/updating table '{full_name}'")
            try:
                self.spark.sql(ddl)
                logger.info(f"Created table '{full_name}'")
            except Exception as e:
                logger.error(f"Failed to create table '{full_name}': {e}")
                raise

    def _get_table_definitions(self) -> dict[str, str]:
        """
        Get DDL statements for all Verdict tables.

        Returns:
            Dictionary mapping table name to DDL statement.
        """
        return {
            "raw.prompt_datasets": f"""
                CREATE TABLE IF NOT EXISTS {self.catalog_name}.raw.prompt_datasets (
                    prompt_id STRING NOT NULL,
                    dataset_version STRING NOT NULL,
                    prompt STRING NOT NULL,
                    ground_truth STRING,
                    metadata MAP<STRING, STRING>,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
                USING DELTA
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact' = 'true',
                    'delta.columnMapping.mode' = 'name',
                    'delta.minReaderVersion' = '2',
                    'delta.minWriterVersion' = '5'
                )
                COMMENT 'Versioned prompts with ground truth for evaluation'
            """,

            "raw.model_responses": f"""
                CREATE TABLE IF NOT EXISTS {self.catalog_name}.raw.model_responses (
                    response_id STRING NOT NULL,
                    prompt_id STRING NOT NULL,
                    model_version STRING NOT NULL,
                    endpoint_name STRING NOT NULL,
                    response STRING,
                    latency_ms DOUBLE,
                    status STRING,
                    error_message STRING,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    run_id STRING
                )
                USING DELTA
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact' = 'true',
                    'delta.columnMapping.mode' = 'name',
                    'delta.minReaderVersion' = '2',
                    'delta.minWriterVersion' = '5'
                )
                COMMENT 'Inference outputs from model endpoints with metadata'
            """,

            "evaluated.eval_results": f"""
                CREATE TABLE IF NOT EXISTS {self.catalog_name}.evaluated.eval_results (
                    eval_id STRING NOT NULL,
                    response_id STRING NOT NULL,
                    prompt_id STRING NOT NULL,
                    model_version STRING NOT NULL,
                    run_id STRING NOT NULL,
                    metric_name STRING NOT NULL,
                    metric_value DOUBLE,
                    metric_details STRING,
                    evaluator STRING,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
                USING DELTA
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact' = 'true',
                    'delta.columnMapping.mode' = 'name',
                    'delta.minReaderVersion' = '2',
                    'delta.minWriterVersion' = '5'
                )
                COMMENT 'Per-response metric scores from evaluation'
            """,

            "metrics.metric_summary": f"""
                CREATE TABLE IF NOT EXISTS {self.catalog_name}.metrics.metric_summary (
                    summary_id STRING NOT NULL,
                    run_id STRING NOT NULL,
                    model_version STRING NOT NULL,
                    dataset_version STRING NOT NULL,
                    metric_name STRING NOT NULL,
                    metric_mean DOUBLE,
                    metric_std DOUBLE,
                    metric_p50 DOUBLE,
                    metric_p95 DOUBLE,
                    sample_size INT,
                    verdict STRING,
                    verdict_reason STRING,
                    baseline_version STRING,
                    baseline_mean DOUBLE,
                    p_value DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
                USING DELTA
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact' = 'true',
                    'delta.columnMapping.mode' = 'name',
                    'delta.minReaderVersion' = '2',
                    'delta.minWriterVersion' = '5'
                )
                COMMENT 'Aggregated metrics per model version and run'
            """
        }

    def run_setup(self) -> None:
        """Run complete catalog setup."""
        logger.info("Starting Verdict catalog setup...")
        self.create_catalog()
        self.create_schemas()
        self.create_tables()
        logger.info("Verdict catalog setup complete!")

    def drop_all(self) -> None:
        """Drop all Verdict catalog objects (use with caution!)."""
        logger.warning(f"Dropping catalog '{self.catalog_name}' and all its objects...")
        try:
            self.spark.sql(f"DROP CATALOG IF EXISTS {self.catalog_name} CASCADE")
            logger.info(f"Catalog '{self.catalog_name}' dropped")
        except Exception as e:
            logger.error(f"Failed to drop catalog: {e}")
            raise


def main() -> None:
    """Main entry point for catalog setup."""
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Verdict Unity Catalog")
    parser.add_argument(
        "--catalog",
        default="verdict",
        help="Catalog name (default: verdict)"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop catalog before creating (WARNING: destroys all data)"
    )
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    setup = VerdictCatalogSetup(catalog_name=args.catalog, spark=spark)

    if args.drop:
        setup.drop_all()

    setup.run_setup()


if __name__ == "__main__":
    main()
