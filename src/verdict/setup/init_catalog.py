"""
Unity Catalog initialization for Verdict.

Creates the catalog, schemas, and Delta tables required for the
LLMOps evaluation framework.

Can be run as a Databricks notebook or Python module.
"""

import logging
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import CreateTable
from databricks.sdk.errors import NotFound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerdictCatalogSetup:
    """Manages Unity Catalog setup for Verdict."""

    def __init__(self, catalog_name: str = "verdict", ws: Optional[WorkspaceClient] = None):
        """
        Initialize catalog setup.

        Args:
            catalog_name: Name of the Unity Catalog catalog.
            ws: Databricks WorkspaceClient instance (creates one if not provided).
        """
        self.catalog_name = catalog_name
        self.ws = ws or WorkspaceClient()
        self.schemas = ["raw", "evaluated", "metrics"]

    def create_catalog(self) -> None:
        """Create the verdict catalog if it doesn't exist."""
        try:
            self.ws.catalogs.get(self.catalog_name)
            logger.info(f"Catalog '{self.catalog_name}' already exists")
        except NotFound:
            logger.info(f"Creating catalog '{self.catalog_name}'")
            self.ws.catalogs.create(
                name=self.catalog_name,
                comment="Verdict LLMOps Evaluation Framework"
            )
            logger.info(f"Created catalog '{self.catalog_name}'")

    def create_schemas(self) -> None:
        """Create schemas within the catalog."""
        for schema_name in self.schemas:
            full_name = f"{self.catalog_name}.{schema_name}"
            try:
                self.ws.schemas.get(full_name)
                logger.info(f"Schema '{full_name}' already exists")
            except NotFound:
                logger.info(f"Creating schema '{full_name}'")
                self.ws.schemas.create(
                    name=schema_name,
                    catalog_name=self.catalog_name,
                    comment=f"Verdict {schema_name} data"
                )
                logger.info(f"Created schema '{full_name}'")

    def create_tables(self) -> None:
        """Create Delta tables for Verdict."""
        tables = self._get_table_definitions()

        for table_name, ddl in tables.items():
            full_name = f"{self.catalog_name}.{table_name}"
            logger.info(f"Creating/updating table '{full_name}'")
            try:
                self.ws.statement_execution.execute_statement(
                    statement=ddl,
                    warehouse_id=self._get_warehouse_id()
                )
                logger.info(f"Created table '{full_name}'")
            except Exception as e:
                logger.error(f"Failed to create table '{full_name}': {e}")
                raise

    def _get_warehouse_id(self) -> str:
        """Get or create a SQL warehouse for DDL execution."""
        warehouses = list(self.ws.warehouses.list())
        if not warehouses:
            raise RuntimeError("No SQL warehouses available. Please create one.")
        # Use the first running warehouse
        for wh in warehouses:
            if wh.state == "RUNNING":
                return wh.id
        return warehouses[0].id

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


def main() -> None:
    """Main entry point for catalog setup."""
    import argparse

    parser = argparse.ArgumentParser(description="Initialize Verdict Unity Catalog")
    parser.add_argument(
        "--catalog",
        default="verdict",
        help="Catalog name (default: verdict)"
    )
    args = parser.parse_args()

    setup = VerdictCatalogSetup(catalog_name=args.catalog)
    setup.run_setup()


if __name__ == "__main__":
    # Support running as Databricks notebook
    try:
        from dbutils import DBUtils  # type: ignore
        dbutils = DBUtils()
    except ImportError:
        dbutils = None  # Running as module, not notebook

    main()