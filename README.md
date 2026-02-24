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
- **RAG Test Dataset Generator**: Generate synthetic Q&A pairs from Qdrant vector database for RAG evaluation

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Prompt Dataset  │ ──▶ │ Inference Runner │ ──▶ │ Model Responses │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        ▲                                                │
        │                                                ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Testgen      │     │ Metric Summary   │ ◀── │ Regr. Detector  │
│ (Qdrant → Q&A)  │     └─────────────────┘     └─────────────────┘
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

# Copy example config
cp .databrickscfg.example ~/.databrickscfg

# Edit with your workspace URL
# Or set environment variables:
export DATABRICKS_HOST="https://adb-<workspace-id>.<random>.azuredatabricks.net"
```

### 2. Configure Authentication

**Option A: Azure AD Service Principal (Recommended)**
```bash
# Create a Service Principal in Azure AD
az ad sp create-for-rbac --name "verdict-sp" --role Contributor

# Set environment variables
export AZURE_TENANT_ID="<tenant-id>"
export AZURE_CLIENT_ID="<client-id>"
export AZURE_CLIENT_SECRET="<client-secret>"
```

**Option B: Personal Access Token**
```bash
# Generate PAT from Databricks workspace -> User Settings -> Developer
export DATABRICKS_TOKEN="dapi..."
```

### 3. Deploy to Azure Databricks

```bash
# Make setup script executable
chmod +x setup_azure.sh

# Run setup (validates config, builds package, deploys)
./setup_azure.sh
```

Or manually:
```bash
# Build the wheel
pip install build
python -m build --wheel

# Deploy using Databricks Asset Bundles
databricks bundle deploy -t development
```

### 4. Create Sample Dataset

Run the sample dataset notebook:
1. Go to **Workspace** → **verdict_dev** → **notebooks**
2. Open `create_sample_dataset.py`
3. Click **Run All**

### 5. Run the Pipeline

```bash
# Run via CLI
databricks bundle run verdict_pipeline -t development
```

Or via Databricks UI:
1. Go to **Workflows**
2. Find "Verdict: LLM Evaluation Pipeline"
3. Click **Run Now**
4. Set parameters:
   - `model_endpoint`: Your model serving endpoint name
   - `candidate_version`: Version to evaluate (e.g., "2")
   - `baseline_version`: Version to compare against (e.g., "1")
   - `dataset_version`: Dataset version (e.g., "v1")

## Project Structure

```
verdict/
├── databricks.yml               # Databricks Asset Bundle config
├── notebooks/                   # Databricks notebooks
│   ├── init_catalog.py          # Unity Catalog setup
│   ├── create_sample_dataset.py # Sample data creation
│   ├── run_testgen.py           # RAG test dataset generation
│   ├── run_inference.py         # Inference task
│   ├── run_evaluation.py        # Evaluation task
│   ├── run_regression.py        # Regression detection
│   └── send_alert.py            # Alert task
├── src/verdict/                 # Python package
│   ├── setup/                   # Unity Catalog setup
│   ├── data/                    # Prompt dataset management
│   ├── inference/               # Parallel inference
│   ├── evaluation/              # Evaluation modules
│   ├── testgen/                 # RAG test dataset generator
│   └── regression/              # Regression detection
├── config/config.yaml           # Configuration
├── dashboard/                   # SQL dashboard queries
└── orchestration/               # Workflow definitions
```

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
- Toxicity

### Custom LLM-as-a-Judge
- 1-5 quality score with reasoning
- Configurable judge model endpoint

### Deterministic
- ROUGE-L
- Exact Match
- Response Length
- Latency (p50/p95)

## Requirements

- Azure Databricks Runtime 14.0+
- Unity Catalog enabled
- Model Serving endpoints deployed
- Python 3.10+
- Azure AD authentication (recommended) or PAT token

## Azure Authentication

Verdict supports multiple authentication methods for Azure Databricks:

### 1. Azure AD Service Principal (Recommended)
```bash
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export DATABRICKS_HOST="https://adb-<workspace-id>.<random>.azuredatabricks.net"
```

### 2. Managed Identity
Automatically used when running on Azure VMs, AKS, or Azure Functions.

### 3. PAT Token
```bash
export DATABRICKS_TOKEN="your-personal-access-token"
export DATABRICKS_HOST="https://adb-<workspace-id>.<random>.azuredatabricks.net"
```

## RAG Test Dataset Generator

The `verdict-testgen` module generates synthetic test datasets from Qdrant vector database for RAG evaluation. It creates Q&A pairs with ground truth answers from your document chunks.

### Installation

```bash
pip install verdict[testgen]
```

### Usage

**CLI:**
```bash
# Set required environment variables
export QDRANT_COLLECTION="documents"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"

# Generate Q&A pairs
verdict-testgen --collection documents --limit 100

# Generate and load to Unity Catalog
verdict-testgen --collection documents --load-to-catalog --dataset-version v1
```

**Python API:**
```python
from verdict.testgen import Settings, TestDatasetGenerator

settings = Settings(
    qdrant_collection="documents",
    qdrant_url="http://localhost:6333",
    qdrant_scroll_limit=100,
)

generator = TestDatasetGenerator(settings)
result = generator.generate()

# Load to Unity Catalog
generator.load_to_catalog(
    qa_pairs=result["qa_pairs_data"],
    version="v1",
    catalog_name="verdict",
)
```

### Output Formats

The generator produces three output files:

| File | Description |
|------|-------------|
| `qa_pairs.jsonl` | Question-answer pairs with chunk references |
| `retrieval_eval.jsonl` | Retrieval evaluation format with hard negatives |
| `rag_eval.jsonl` | RAG evaluation format for end-to-end testing |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `QDRANT_COLLECTION` | Yes | Qdrant collection name |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `QDRANT_URL` | No | Qdrant server URL (default: `http://localhost:6333`) |
| `QDRANT_API_KEY` | No | Qdrant Cloud API key |
| `OUTPUT_DIR` | No | Output directory (default: `./output`) |

## License

MIT