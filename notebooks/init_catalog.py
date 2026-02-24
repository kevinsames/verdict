# Databricks notebook source
# ---
# title: Initialize Unity Catalog
# description: Creates the verdict catalog, schemas, and Delta tables
# ---

# COMMAND ----------
# MAGIC %pip install -e /Workspace/Repos/verdict

# COMMAND ----------
import logging
from verdict.setup.init_catalog import VerdictCatalogSetup
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
# dbutils.widgets.text("catalog_name", "verdict")
# catalog_name = dbutils.widgets.get("catalog_name")
catalog_name = "verdict"

# COMMAND ----------
logger.info(f"Initializing Unity Catalog: {catalog_name}")

setup = VerdictCatalogSetup(catalog_name=catalog_name)
setup.run_setup()

# COMMAND ----------
logger.info("Unity Catalog setup complete!")

# Verify tables exist
spark.sql(f"SHOW TABLES IN {catalog_name}.raw").display()
spark.sql(f"SHOW TABLES IN {catalog_name}.evaluated").display()
spark.sql(f"SHOW TABLES IN {catalog_name}.metrics").display()
