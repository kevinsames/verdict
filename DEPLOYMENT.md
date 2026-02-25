# Azure Databricks Deployment Guide

## Prerequisites

1. **Azure Databricks Workspace** with Unity Catalog enabled
2. **Databricks CLI** installed:
   ```bash
   pip install databricks-cli
   ```
3. **Azure CLI** (for Azure AD authentication):
   ```bash
   brew install azure-cli
   ```

## Step 1: Configure Authentication

### Option A: Azure AD Service Principal (Recommended for CI/CD)

```bash
# Set environment variables
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export DATABRICKS_HOST="https://adb-<workspace-id>.<random>.azuredatabricks.net"
```

### Option B: Personal Access Token (for development)

```bash
# Generate PAT from Databricks UI: User Settings > Developer > Access Tokens
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_HOST="https://adb-<workspace-id>.<random>.azuredatabricks.net"
```

## Step 2: Deploy Verdict

```bash
# Deploy using Databricks Asset Bundles
# Note: Uses git_source - no whl build required
databricks bundle deploy -t development
```

The bundle uses `git_source` to pull code directly from GitHub, so no wheel file needs to be built.

## Step 3: Initialize Unity Catalog

1. Go to your Azure Databricks workspace
2. Navigate to **Workflows** > **verdict_pipeline**
3. Run the `init_catalog` task

This creates the `verdict_dev` catalog with schemas:
- `raw` - Source data (prompt_datasets, model_responses)
- `evaluated` - Evaluation results
- `metrics` - Aggregated metrics

**Note:** If catalog creation fails (e.g., requires managed location), create the catalog manually via Databricks UI, then re-run init_catalog to create schemas and tables.

## Step 4: Create Sample Dataset

Run the `notebooks/create_sample_dataset.ipynb` notebook to create a test dataset.

Or create your own dataset:

```python
from verdict.data.prompt_dataset import PromptDatasetManager

manager = PromptDatasetManager(catalog_name="verdict_dev")
manager.create_dataset(
    prompts=[
        {"prompt": "Your question?", "ground_truth": "Expected answer"}
    ],
    version="v1"
)
```

## Step 5: Deploy Your Model

1. Register your model in MLflow Model Registry
2. Create a Model Serving endpoint via Databricks UI: **Compute** → **Serving** → **Create serving endpoint**

Or via CLI:
```bash
databricks api post /api/2.0/serving-endpoints --json '{
  "name": "my-model-endpoint",
  "config": {
    "served_models": [{
      "model_name": "my-model",
      "model_version": "1",
      "scale_to_zero_enabled": true,
      "workload_size": "Small"
    }]
  }
}'
```

## Step 6: Run the Pipeline

```bash
# Run with default parameters
databricks bundle run verdict_pipeline -t development

# Run with custom parameters
databricks bundle run verdict_pipeline -t development \
  --var model_endpoint=my-model-endpoint \
  --var candidate_version=2 \
  --var baseline_version=1
```

## Step 7: View Results

### MLflow Experiments
Navigate to **Experiments** > `/verdict/experiments`

### Metric Tables
```sql
-- View verdict history
SELECT * FROM verdict_dev.metrics.metric_summary ORDER BY created_at DESC;

-- View detailed results
SELECT * FROM verdict_dev.evaluated.eval_results WHERE run_id = '<run_id>';
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_endpoint` | Model Serving endpoint to evaluate | `databricks-gpt-oss-20b` |
| `judge_endpoint` | LLM judge model for evaluation | `databricks-llama-4-maverick` |
| `catalog_name` | Unity Catalog name | `verdict_dev` |
| `threshold_pct` | Regression threshold percentage | `5.0` |
| `p_value_threshold` | Statistical significance level | `0.05` |

## Data Operations

Verdict uses **PySpark directly** for all Unity Catalog and Delta Lake operations. No databricks-sdk is required for data operations - everything uses native Spark SQL through `spark.sql()`.

## Troubleshooting

### Authentication Errors
```bash
# Test connection
databricks clusters list
```

### Unity Catalog Permission Errors
Ensure your user/service principal has:
- `CREATE CATALOG` permission on metastore
- `CREATE SCHEMA` permission on catalog

### Model Serving Not Found
Verify your endpoint is deployed and running:
```bash
databricks serving-endpoints get my-model-endpoint
```

## Next Steps

1. Create Azure Key Vault secrets for webhook URLs
2. Schedule the pipeline to run automatically
3. Create Lakeview dashboards for monitoring