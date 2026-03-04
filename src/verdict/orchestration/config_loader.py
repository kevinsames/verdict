"""
Configuration loader for Verdict pipeline.

Loads and merges YAML configuration files for different environments.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config directory (relative to this file)
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"

# Environment names
ENV_DEVELOPMENT = "development"
ENV_STAGING = "staging"
ENV_PRODUCTION = "production"


def get_environment() -> str:
    """
    Get the current environment name.

    Environment is determined by:
    1. VERDICT_ENV environment variable
    2. DATABRICKS_ENV environment variable
    3. Default to 'development'

    Returns:
        Environment name (development/staging/production).
    """
    return (
        os.environ.get("VERDICT_ENV")
        or os.environ.get("DATABRICKS_ENV")
        or ENV_DEVELOPMENT
    )


def load_config_file(file_path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Configuration dictionary.
    """
    if not file_path.exists():
        logger.warning(f"Config file not found: {file_path}")
        return {}

    with open(file_path) as f:
        config = yaml.safe_load(f) or {}

    logger.debug(f"Loaded config from {file_path}")
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Values in override take precedence over base.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Deep merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    environment: str | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load and merge configuration for the given environment.

    Loads in order (later overrides earlier):
    1. base.yaml - shared defaults
    2. {environment}.yaml - environment-specific overrides

    Args:
        environment: Environment name (development/staging/production).
                     If None, uses get_environment().
        config_dir: Directory containing config files.
                    If None, uses DEFAULT_CONFIG_DIR.

    Returns:
        Merged configuration dictionary.
    """
    env = environment or get_environment()
    config_path = config_dir or DEFAULT_CONFIG_DIR

    logger.info(f"Loading config for environment: {env}")
    logger.debug(f"Config directory: {config_path}")

    # Load base configuration
    base_file = config_path / "base.yaml"
    base_config = load_config_file(base_file)

    # Load environment-specific configuration
    env_file = config_path / f"{env}.yaml"
    env_config = load_config_file(env_file)

    # Merge configurations
    config = merge_configs(base_config, env_config)

    # Add environment metadata
    config["_environment"] = env
    config["_config_dir"] = str(config_path)

    logger.info(f"Configuration loaded for environment: {env}")
    return config


def get_config_value(
    config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        config: Configuration dictionary.
        key: Dot-notation key (e.g., "endpoints.model").
        default: Default value if key not found.

    Returns:
        Configuration value or default.
    """
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def validate_config(config: dict[str, Any]) -> list[str]:
    """
    Validate required configuration values.

    Args:
        config: Configuration dictionary.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Check for required top-level keys
    required_keys = ["catalog_name", "model_endpoint", "judge_endpoint"]
    for key in required_keys:
        if key not in config or not config[key]:
            errors.append(f"Missing required config: {key}")

    return errors


def get_default_config() -> dict[str, Any]:
    """
    Get default configuration for development.

    This is useful for local testing or when config files are not available.

    Returns:
        Default configuration dictionary.
    """
    return {
        "catalog_name": "verdict_dev",
        "model_endpoint": "databricks-gpt-oss-20b",
        "judge_endpoint": "databricks-llama-4-maverick",
        "baseline_version": "1",
        "candidate_version": "2",
        "dataset_version": "v1",
        "threshold": 5.0,
        "p_value": 0.05,
        "evaluation": {
            "sample_size": None,
            "batch_size": 100,
            "metrics": ["faithfulness", "answer_relevance", "toxicity"],
        },
        "regression": {
            "threshold_pct": 5.0,
            "p_value_threshold": 0.05,
            "metrics": ["faithfulness", "answer_relevance", "judge_score", "rouge_l", "latency_p95"],
        },
        "alerts": {
            "webhook_secret": "verdict/alerts_webhook",
            "on_labels": ["WARN", "FAIL"],
        },
        "testgen": {
            "qdrant_collection": "documents",
            "qdrant_url": "http://localhost:6333",
            "limit": 200,
        },
    }