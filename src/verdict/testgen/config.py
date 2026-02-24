"""
Configuration settings for RAG Test Dataset Generator.

Uses pydantic-settings to load configuration from environment variables.
"""

import logging

from pydantic import Field
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Configuration settings for the RAG Test Dataset Generator.

    All settings can be overridden via environment variables.
    """

    # Qdrant settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant connection URL"
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant Cloud API key (optional)"
    )
    qdrant_collection: str = Field(
        ...,
        description="Qdrant collection name (required)"
    )
    qdrant_text_field: str = Field(
        default="text",
        description="Payload field containing chunk text"
    )
    qdrant_scroll_limit: int = Field(
        default=200,
        description="Max chunks to process"
    )

    # Azure OpenAI settings
    azure_openai_endpoint: str = Field(
        ...,
        description="Azure OpenAI resource URL (required)"
    )
    azure_openai_api_key: str = Field(
        ...,
        description="Azure OpenAI API key (required)"
    )
    azure_openai_deployment: str = Field(
        default="gpt-4o",
        description="Model deployment name"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01",
        description="API version"
    )

    # Generation settings
    samples_per_chunk: int = Field(
        default=1,
        description="Q&A pairs generated per chunk"
    )
    hard_negatives_per_query: int = Field(
        default=2,
        description="Hard negatives per query"
    )
    output_dir: str = Field(
        default="./output",
        description="Output directory path"
    )
    delay_between_calls: float = Field(
        default=0.5,
        description="Rate limit buffer in seconds"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
