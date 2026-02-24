# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Verdict** is an automated LLMOps Evaluation Framework that evaluates LLM outputs at scale on Databricks. It tracks quality metrics over time, detects regressions across model versions, and triggers alerts.

## Tech Stack

- Delta Lake + Unity Catalog (storage & governance)
- Databricks Model Serving (inference)
- MLflow LLM Evaluate + custom LLM-as-a-judge (evaluation)
- Databricks Workflows (orchestration)
- Databricks SQL / Lakeview (dashboards)
- Python + PySpark

## Commands

### Running Tests
```bash
pytest tests/ -v
pytest tests/ -v -k "test_name"  # Run specific test
```

### Type Checking
```bash
mypy src/
```

### Linting
```bash
ruff check src/
ruff format src/
```

## Architecture

### Unity Catalog Structure
- **Catalog**: `verdict`
- **Schemas**: `raw`, `evaluated`, `metrics`
- **Tables**:
  - `raw.prompt_datasets` — versioned prompts with ground truth
  - `raw.model_responses` — inference outputs with metadata
  - `evaluated.eval_results` — per-response metric scores
  - `metrics.metric_summary` — aggregated metrics per model version + run

### Data Flow
```
prompt_datasets → inference_runner → model_responses → evaluator → eval_results → regression_detector → metric_summary
```

### Evaluation Pipeline
1. **Inference**: Spark UDFs hit Model Serving REST API in parallel
2. **Evaluation**: MLflow LLM Evaluate + custom LLM-as-a-judge scorers
3. **Regression Detection**: Mann-Whitney U test compares candidate vs baseline
4. **Verdict Delivery**: Alert on regression (email/webhook)

### Verdict Labels
- `PASS` — No regression detected
- `WARN` — Minor metric drop (within threshold)
- `FAIL` — Significant regression detected

## Code Standards

- **Delta writes**: Use `MERGE` for idempotency
- **Type hints**: Required on all functions
- **Docstrings**: Required on all functions
- **API calls**: Use `databricks-sdk` where possible
- **Secrets**: Reference via `dbutils.secrets.get()` — never hardcode
- **Dual execution**: All scripts must run as both Databricks notebook and Python module
- **Logging**: Use Python's `logging` module throughout

## Configuration

Central config in `config/config.yaml`:
- Catalog/schema names
- Model endpoint URLs
- Judge model endpoint
- Regression thresholds per metric
- MLflow experiment path
- Alert webhook URL