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
# Run the setup script
chmod +x setup_azure.sh
./setup_azure.sh
```

Or manually:

```bash
# Build the wheel
pip install build
python -m build --wheel

# Validate configuration
databricks bundle validate -t development

# Deploy to Azure Databricks
databricks bundle deploy -t development
```

## Step 3: Initialize Unity Catalog

1. Go to your Azure Databricks workspace
2. Navigate to **Workflows** > **verdict_pipeline**
3. Run the `init_catalog` task (or run the full pipeline)

Or run the notebook directly:
- Open `notebooks/init_catalog.py` in Databricks
- Run all cells to create catalog, schemas, and tables

## Step 4: Create Sample Dataset

Run the `notebooks/create_sample_dataset.py` notebook to create a test dataset.

Or create your own dataset:

```python
from verdict.data.prompt_dataset import PromptDatasetManager

manager = PromptDatasetManager(catalog_name="verdict")
manager.create_dataset(
    prompts=[
        {"prompt": "Your question?", "ground_truth": "Expected answer"}
    ],
    version="v1"
)
```

## Step 5: Deploy Your Model

1. Register your model in MLflow Model Registry
2. Create a Model Serving endpoint:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

w = WorkspaceClient()
w.serving_endpoints.create(
    name="my-model-endpoint",
    config=EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                model_name="my-model",
                model_version="1",
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
)
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

### Dashboard
Run the SQL queries in `dashboard/verdict_dashboard.sql` to create Lakeview dashboards

### Metric Tables
```sql
-- View verdict history
SELECT * FROM verdict.metrics.metric_summary ORDER BY created_at DESC;

-- View detailed results
SELECT * FROM verdict.evaluated.eval_results WHERE run_id = '<run_id>';
```

## Configuration Options

Edit `config/config.yaml` or set parameters in the workflow:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_endpoint` | Model Serving endpoint to evaluate | `your-model-endpoint` |
| `judge_endpoint` | LLM judge model for evaluation | `databricks-llama-4` |
| `threshold_pct` | Regression threshold percentage | `5.0` |
| `p_value_threshold` | Statistical significance level | `0.05` |

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
2. Set up alerting in `config/config.yaml`
3. Schedule the pipeline to run automatically
4. Create Lakeview dashboards for monitoring