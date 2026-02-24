"""
LLM-based Q&A synthesizer using LangChain and Azure OpenAI.

Generates question-answer pairs from document chunks.
"""

import logging
import time
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

from verdict.testgen.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum retry attempts for LLM calls
MAX_RETRIES = 3
# Initial backoff in seconds
INITIAL_BACKOFF = 1.0


class QAPair(BaseModel):
    """A single question-answer pair."""

    question: str
    answer: str


class QAPairList(BaseModel):
    """A list of question-answer pairs."""

    pairs: list[QAPair]


# Prompt template for Q&A generation
QA_GENERATION_PROMPT = """You are an expert at creating high-quality question-answer pairs for RAG evaluation.

Given the following context text, generate {num_pairs} question-answer pairs that a real user might ask.

Rules:
1. Questions should be diverse (factual, comparative, how-to, inferential)
2. Answers must be grounded STRICTLY in the provided context - do not add information
3. Questions should be natural and conversational
4. Answers should be complete but concise

Context:
{context}

{format_instructions}

Generate {num_pairs} question-answer pairs now."""


class QASynthesizer:
    """
    Synthesizes Q&A pairs from document chunks using Azure OpenAI.

    Args:
        settings: Configuration settings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment_name=settings.azure_openai_deployment,
            temperature=0.3,
        )
        self.parser = PydanticOutputParser(pydantic_object=QAPairList)

        self.prompt = ChatPromptTemplate.from_template(
            QA_GENERATION_PROMPT
        )

        self.chain = self.prompt | self.llm | self.parser

    def generate_qa_pairs(
        self,
        chunk_text: str,
        chunk_id: str,
    ) -> list[dict[str, Any]]:
        """
        Generate Q&A pairs from a document chunk.

        Args:
            chunk_text: The text content of the chunk.
            chunk_id: The ID of the chunk.

        Returns:
            List of dicts with question, answer, and metadata.

        Raises:
            ValueError: If chunk text is too short.
            RuntimeError: If LLM fails after all retries.
        """
        if len(chunk_text) < 50:
            raise ValueError(f"Chunk {chunk_id} is too short ({len(chunk_text)} chars)")

        num_pairs = self.settings.samples_per_chunk
        format_instructions = self.parser.get_format_instructions()

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                result = self.chain.invoke({
                    "context": chunk_text,
                    "num_pairs": num_pairs,
                    "format_instructions": format_instructions,
                })

                # Add delay between calls to respect rate limits
                if self.settings.delay_between_calls > 0:
                    time.sleep(self.settings.delay_between_calls)

                # Transform to output format
                qa_pairs = []
                for pair in result.pairs:
                    qa_pairs.append({
                        "chunk_id": chunk_id,
                        "question": pair.question,
                        "ground_truth": pair.answer,
                        "source_text": chunk_text,
                    })

                logger.debug(f"Generated {len(qa_pairs)} Q&A pairs for chunk {chunk_id}")
                return qa_pairs

            except Exception as e:
                last_error = e
                wait_time = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed for chunk {chunk_id}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to generate Q&A pairs for chunk {chunk_id} after {MAX_RETRIES} attempts: "
            f"{last_error}"
        )
