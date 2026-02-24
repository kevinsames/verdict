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

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Prompt Dataset  │ ──▶ │ Inference Runner │ ──▶ │ Model Responses │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Metric Summary  │ ◀── │ Regr. Detector   │ ◀── │  Evaluator      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│    Dashboard    │     │     Alerts       │
└─────────────────┘     └──────────────────┘
```

## Quick Start

### 1. Setup Unity Catalog

```bash
python setup/init_catalog.py
```

### 2. Configure

Update `config/config.yaml` with your endpoint names and thresholds.

### 3. Run Inference

```bash
python inference/inference_runner.py \
  --endpoint my-model-endpoint \
  --dataset prompts_v1
```

### 4. Run Evaluation

```bash
python evaluation/mlflow_evaluator.py \
  --responses-table raw.model_responses \
  --run-id <run_id>
```

### 5. Check for Regressions

```bash
python regression/regression_detector.py \
  --candidate-version 2 \
  --baseline-version 1
```

## Project Structure

```
verdict/
├── README.md
├── CLAUDE.md
├── setup/
│   └── init_catalog.py          # Unity Catalog setup
├── data/
│   └── prompt_dataset.py        # Prompt/ground-truth management
├── inference/
│   └── inference_runner.py      # Parallel inference
├── evaluation/
│   ├── mlflow_evaluator.py      # MLflow LLM Evaluate
│   ├── custom_judges.py         # LLM-as-a-judge
│   └── deterministic_metrics.py # ROUGE, exact match
├── regression/
│   └── regression_detector.py   # Statistical comparison
├── orchestration/
│   └── verdict_workflow.yaml    # Databricks Workflow
├── dashboard/
│   └── verdict_dashboard.sql    # Dashboard queries
└── config/
    └── config.yaml              # Centralized config
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

## License

MIT