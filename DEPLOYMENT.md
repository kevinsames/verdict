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
databricks bundle deploy -t development
```

The bundle uses `git_source` to pull code directly from GitHub:
- No wheel file needs to be built
- Notebooks add `src/` to `sys.path` for imports
- Changes are picked up by re-running the job

## Step 3: Initialize Unity Catalog

Run the `init_catalog` notebook from the `verdict_pipeline` job.

This creates:
- **Catalog**: `verdict_dev`
- **Schemas**: `raw`, `evaluated`, `metrics`
- **Tables**: `prompt_datasets`, `model_responses`, `eval_results`, `metric_summary`
- **Volume**: `raw.testgen_output` (for RAG test dataset generator output)

**Troubleshooting:**
If catalog creation fails with "storage location required", create the catalog manually:
1. Go to **Catalog** → **Create Catalog**
2. Provide a managed location
3. Re-run `init_catalog` to create schemas and tables

## Step 4: Create Sample Dataset

Run `notebooks/create_sample_dataset.ipynb` to create test data.

Or programmatically:
```python
from verdict.data.prompt_dataset import PromptDatasetManager

manager = PromptDatasetManager(catalog_name="verdict_dev")
manager.create_dataset(
    prompts=[{"prompt": "Question?", "ground_truth": "Answer"}],
    version="v1"
)
```

## Step 5: Deploy Your Model

1. Register model in MLflow Model Registry
2. Create Model Serving endpoint via **Compute** → **Serving**

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
# Run with defaults
databricks bundle run verdict_pipeline -t development

# Run with custom parameters
databricks bundle run verdict_pipeline -t development \
  --var model_endpoint=my-model-endpoint \
  --var candidate_version=2 \
  --var baseline_version=1
```

## Step 7: View Results

### SQL Queries
```sql
-- Verdict history
SELECT * FROM verdict_dev.metrics.metric_summary ORDER BY created_at DESC;

-- Detailed results
SELECT * FROM verdict_dev.evaluated.eval_results WHERE run_id = '<run_id>';

-- Model responses
SELECT * FROM verdict_dev.raw.model_responses WHERE run_id = '<run_id>';
```

### MLflow Experiments
Navigate to **Experiments** → `/verdict/experiments`

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `catalog_name` | Unity Catalog name | `verdict_dev` |
| `model_endpoint` | Model Serving endpoint | `databricks-gpt-oss-20b` |
| `judge_endpoint` | LLM judge model | `databricks-llama-4-maverick` |
| `dataset_version` | Prompt dataset version | `v1` |
| `threshold_pct` | Regression threshold % | `5.0` |
| `p_value_threshold` | Statistical significance | `0.05` |

## Data Operations

Verdict uses **PySpark directly** for all data operations:
- Unity Catalog: `spark.sql("CREATE CATALOG IF NOT EXISTS ...")`
- Delta Lake: `spark.sql("CREATE TABLE IF NOT EXISTS ... USING DELTA")`
- No databricks-sdk required for data operations

## Deployment Targets

| Target | Catalog | Root Path |
|--------|---------|-----------|
| `development` | `verdict_dev` | `/Users/{user}/verdict_dev` |
| `staging` | `verdict_staging` | `/Shared/verdict_staging` |
| `production` | `verdict` | `/Shared/verdict` |

## Troubleshooting

### Authentication Errors
```bash
databricks clusters list
```

### Unity Catalog Permissions
Ensure your user/service principal has:
- `CREATE CATALOG` on metastore
- `CREATE SCHEMA` on catalog

### Model Serving Not Found
```bash
databricks serving-endpoints get my-model-endpoint
```

### Notebook Import Errors
Notebooks add `src/` to `sys.path`. If imports fail, verify:
1. Job uses `git_source` pointing to correct branch
2. Working directory is the repo root

## Next Steps

1. Configure webhook secrets in Azure Key Vault
2. Schedule pipeline runs
3. Build Lakeview dashboards