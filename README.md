# Verdict

**Production-grade LLMOps Evaluation Framework on Azure Databricks**

Verdict is an automated system that evaluates LLM outputs at scale, tracks quality metrics over time, detects regressions across model versions, and triggers alerts — all natively on Azure Databricks.

## Features

- **Scalable Inference**: Spark-based parallel inference against Azure Databricks Model Serving endpoints
- **Multi-metric Evaluation**: MLflow LLM Evaluate + custom LLM-as-a-judge scorers
- **Regression Detection**: Statistical comparison across model versions with Mann-Whitney U test
- **Automated Alerts**: Email/webhook notifications on quality regressions
- **Full Governance**: Unity Catalog integration with Delta Lake storage
- **Azure Native**: Supports Azure AD, Managed Identity, and Azure Key Vault integration
- **RAG Test Dataset Generator**: Generate synthetic Q&A pairs from Qdrant vector database
- **Run Context Pattern**: Centralized configuration and state via `metadata.pipeline_runs` table

## Architecture

### Run Context Pattern

All tasks read from and write to `metadata.pipeline_runs` instead of passing parameters:

```python
# In any notebook - get configuration
from verdict.orchestration import RunContext

ctx = RunContext.from_run_id(spark, catalog_name, run_id)
config = ctx.get_config()

model_endpoint = config["model_endpoint"]
candidate_version = config["candidate_version"]

# Store outputs for downstream tasks
ctx.set_output("inference_run_id", inference_run_id)
ctx.update_task_status("inference", "completed")
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     metadata.pipeline_runs                          │
│              (Central run state & configuration)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   init_run      │ ──▶ │  init_catalog    │ ──▶ │   Inference     │
│ (load config)   │     │  (create tables) │     │   Runner        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
        ┌───────────────────────────────────────────────┤
        │                                               │
        ▼                                               ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Testgen      │     │ Regr. Detector   │ ◀── │   Evaluation    │
│ (Qdrant → Q&A)  │     └──────────────────┘     └─────────────────┘
└─────────────────┘             │                       │
                                ▼                       ▼
                        ┌─────────────────┐     ┌──────────────────┐
                        │    Dashboard    │     │     Alerts       │
                        └─────────────────┘     └──────────────────┘
```

## Quick Start

### Prerequisites

1. **Azure Databricks Workspace** with Unity Catalog enabled
2. **Databricks CLI** installed:
   ```bash
   pip install databricks-cli
   ```
3. **Azure CLI** (for Azure AD authentication):
   ```bash
   brew install azure-cli
   ```

### 1. Clone and Configure

```bash
git clone https://github.com/kevinsames/verdict.git
cd verdict

# Set environment variables
export DATABRICKS_HOST="https://adb-<workspace-id>.<random>.azuredatabricks.net"
```

### 2. Configure Authentication

**Option A: Azure AD Service Principal (Recommended)**
```bash
az ad sp create-for-rbac --name "verdict-sp" --role Contributor

export AZURE_TENANT_ID="<tenant-id>"
export AZURE_CLIENT_ID="<client-id>"
export AZURE_CLIENT_SECRET="<client-secret>"
```

**Option B: Personal Access Token**
```bash
export DATABRICKS_TOKEN="dapi..."
```

### 3. Deploy to Azure Databricks

```bash
# Uses git_source - no wheel build required
databricks bundle deploy -t development
```

### 4. Initialize Unity Catalog

Run the `init_catalog` notebook to create the `verdict_dev` catalog, schemas, and tables.

**Note:** If catalog creation fails (requires managed location), create it manually via Databricks UI, then re-run init_catalog.

### 5. Run the Pipeline

```bash
databricks bundle run verdict_pipeline -t development
```

## Project Structure

```
verdict/
├── config/                      # Environment-specific configs
│   ├── base.yaml               # Shared defaults
│   ├── development.yaml        # Dev environment
│   ├── staging.yaml            # Staging environment
│   └── production.yaml         # Production environment
├── databricks.yml              # Databricks Asset Bundle config
├── orchestration/
│   └── verdict_workflow.yaml   # Job definition
├── notebooks/                  # Databricks notebooks (.ipynb)
│   ├── init_run.ipynb          # Initialize run record
│   ├── init_catalog.ipynb      # Unity Catalog setup
│   ├── run_inference.ipynb     # Inference task
│   ├── run_evaluation.ipynb    # Evaluation task
│   ├── run_regression.ipynb    # Regression detection
│   ├── send_alert.ipynb        # Alert task
│   └── run_testgen.ipynb       # RAG test dataset generation
├── src/verdict/                 # Python package
│   ├── orchestration/          # Run context & config loader
│   ├── setup/                  # Unity Catalog setup
│   ├── data/                   # Prompt dataset management
│   ├── inference/              # Parallel inference
│   ├── evaluation/             # Evaluation modules
│   ├── testgen/                # RAG test dataset generator
│   └── regression/             # Regression detection
├── pyproject.toml              # Python package config
└── requirements.txt            # Dependencies
```

## Unity Catalog Schema

| Schema | Table | Description |
|--------|-------|-------------|
| `raw` | `prompt_datasets` | Versioned prompts with ground truth |
| `raw` | `model_responses` | Inference outputs with metadata |
| `evaluated` | `eval_results` | Per-response metric scores |
| `metrics` | `metric_summary` | Aggregated metrics per model version |
| `metadata` | `pipeline_runs` | Run state, configuration, and outputs |

## Configuration

Configuration is managed via YAML files in `config/`:

```yaml
# config/development.yaml
catalog_name: verdict_dev
model_endpoint: databricks-gpt-oss-20b
judge_endpoint: databricks-llama-4-maverick
baseline_version: "1"
candidate_version: "2"
dataset_version: v1
```

Secrets (webhook URLs, API keys) are stored in Databricks Secret Scopes:
- `verdict/alerts_webhook` - Webhook URL for alerts
- `verdict/qdrant_api_key` - Qdrant API key (for testgen)

## Verdict Labels

| Label | Meaning |
|-------|---------|
| `PASS` | No regression detected, metrics within acceptable range |
| `WARN` | Minor metric drop detected, within warning threshold |
| `FAIL` | Significant regression detected, requires attention |

## Metrics

### Built-in (MLflow LLM Evaluate)
- Faithfulness
- Answer Relevance

### Custom LLM-as-a-Judge
- 1-5 quality score with reasoning
- Configurable judge model endpoint (default: `databricks-llama-4-maverick`)

### Deterministic
- ROUGE-L
- Exact Match
- Response Length
- Latency (p50/p95)

## RAG Test Dataset Generator

Generates synthetic Q&A test datasets from Qdrant vector database.

### Usage

```bash
# Set environment variables
export QDRANT_COLLECTION="documents"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"

# Generate Q&A pairs
verdict-testgen --collection documents --limit 100

# Generate and load to Unity Catalog
verdict-testgen --collection documents --load-to-catalog --dataset-version v1
```

### Output

Files written to `/Volumes/verdict_dev/raw/testgen_output/`:
- `qa_pairs.jsonl` — Question-answer pairs
- `retrieval_eval.jsonl` — Retrieval evaluation format
- `rag_eval.jsonl` — RAG evaluation format

## Requirements

- Azure Databricks Runtime 14.0+
- Unity Catalog enabled
- Model Serving endpoints deployed
- Python 3.10+

## Data Operations

Verdict uses **PySpark directly** for all Unity Catalog and Delta Lake operations. No databricks-sdk is required — everything uses native Spark SQL through `spark.sql()`.

## License

MIT