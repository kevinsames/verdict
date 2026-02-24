# Databricks notebook source
# ---
# title: Run Testgen
# description: Generate RAG test dataset from Qdrant and load to Unity Catalog
# ---

# COMMAND ----------
# MAGIC %pip install /Workspace/Repos/verdict/dist/verdict-*.whl

# COMMAND ----------
import logging
import os
from verdict.testgen import Settings, TestDatasetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
# Widget parameters
dbutils.widgets.text("qdrant_collection", "documents", "Qdrant Collection")
dbutils.widgets.text("qdrant_url", "http://localhost:6333", "Qdrant URL")
dbutils.widgets.text("dataset_version", "v1", "Dataset Version")
dbutils.widgets.text("catalog_name", "verdict", "Catalog Name")
dbutils.widgets.text("limit", "200", "Max Chunks")
dbutils.widgets.text("output_dir", "/tmp/testgen_output", "Output Directory")

qdrant_collection = dbutils.widgets.get("qdrant_collection")
qdrant_url = dbutils.widgets.get("qdrant_url")
dataset_version = dbutils.widgets.get("dataset_version")
catalog_name = dbutils.widgets.get("catalog_name")
limit = int(dbutils.widgets.get("limit") or "200")
output_dir = dbutils.widgets.get("output_dir")

# COMMAND ----------
# Required environment variables (set via cluster spark env vars or secrets)
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
# - QDRANT_API_KEY (optional, for Qdrant Cloud)

logger.info(f"Generating test dataset from Qdrant collection: {qdrant_collection}")
logger.info(f"Qdrant URL: {qdrant_url}")
logger.info(f"Dataset version: {dataset_version}")
logger.info(f"Catalog: {catalog_name}")

# COMMAND ----------
# Validate required environment variables
required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
missing = [v for v in required_vars if not os.environ.get(v)]
if missing:
    raise ValueError(f"Missing required environment variables: {missing}")

# COMMAND ----------
# Create settings
settings = Settings(
    qdrant_collection=qdrant_collection,
    qdrant_url=qdrant_url,
    qdrant_scroll_limit=limit,
    output_dir=output_dir,
)

# COMMAND ----------
# Generate dataset
generator = TestDatasetGenerator(settings)
result = generator.generate()

# COMMAND ----------
# Load to Unity Catalog if Q&A pairs were generated
qa_count = result.get("qa_pairs", 0)
if qa_count > 0:
    logger.info(f"Loading {qa_count} Q&A pairs to Unity Catalog...")
    loaded = generator.load_to_catalog(
        qa_pairs=result["qa_pairs_data"],
        version=dataset_version,
        catalog_name=catalog_name,
    )
    print(f"\nSuccessfully loaded {loaded} prompts to {catalog_name}.raw.prompt_datasets (version: {dataset_version})")
else:
    print("\nNo Q&A pairs generated. Check Qdrant collection and connectivity.")

# COMMAND ----------
# Summary
print(f"\n=== Testgen Summary ===")
print(f"Chunks processed: {result.get('chunks', 0)}")
print(f"Q&A pairs generated: {qa_count}")
print(f"Skipped chunks: {result.get('skipped', 0)}")

# COMMAND ----------
# Return values for downstream tasks
dbutils.jobs.taskValues.set("testgen_qa_count", qa_count)
dbutils.jobs.taskValues.set("testgen_version", dataset_version)