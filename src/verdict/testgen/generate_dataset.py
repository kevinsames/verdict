"""
CLI entrypoint for RAG Test Dataset Generator.

Generates synthetic test datasets from Qdrant vector database.
"""

import argparse
import logging
import sys
from typing import Any

from tqdm import tqdm

from verdict.testgen.config import Settings
from verdict.testgen.exporters import DatasetExporter
from verdict.testgen.qdrant_sampler import QdrantSampler
from verdict.testgen.synthesizer import QASynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatasetGenerator:
    """
    Main class for generating RAG test datasets.

    Orchestrates sampling, synthesis, and export of test data.

    Args:
        settings: Configuration settings.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sampler = QdrantSampler(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            api_key=settings.qdrant_api_key,
            text_field=settings.qdrant_text_field,
            scroll_limit=settings.qdrant_scroll_limit,
        )
        self.synthesizer = QASynthesizer(settings)
        self.exporter = DatasetExporter(settings.output_dir)

    def generate(
        self,
        filter_field: str | None = None,
        filter_value: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Generate RAG test datasets.

        Args:
            filter_field: Optional Qdrant payload field to filter by.
            filter_value: Value to match for the filter field.
            dry_run: If True, only fetch and preview chunks without LLM calls.

        Returns:
            Dict with generation statistics.
        """
        print(f"\nFetching chunks from '{self.settings.qdrant_collection}'...")

        # Fetch chunks from Qdrant
        chunks = self.sampler.fetch_chunks(
            filter_field=filter_field,
            filter_value=filter_value,
        )

        if not chunks:
            logger.error("No chunks found. Exiting.")
            return {"chunks": 0, "qa_pairs": 0, "skipped": 0}

        print(f"  Fetched {len(chunks)} chunks")

        if dry_run:
            print("\n[DRY RUN] Preview of first 3 chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n  [{i+1}] ID: {chunk['id']}")
                print(f"      Text: {chunk['text'][:100]}...")
                print(f"      Metadata: {chunk['metadata']}")
            return {"chunks": len(chunks), "qa_pairs": 0, "skipped": 0, "dry_run": True}

        # Generate Q&A pairs
        print("\nGenerating Q&A pairs...")
        qa_pairs: list[dict[str, Any]] = []
        hard_negatives: dict[str, list[str]] = {}
        skipped = 0

        for chunk in tqdm(chunks, desc="  Processing chunks"):
            try:
                pairs = self.synthesizer.generate_qa_pairs(
                    chunk_text=chunk["text"],
                    chunk_id=chunk["id"],
                )
                qa_pairs.extend(pairs)

                # Generate hard negatives for each unique chunk
                if chunk["id"] not in hard_negatives:
                    hard_negatives[chunk["id"]] = self.sampler.get_hard_negatives(
                        chunks=chunks,
                        source_id=chunk["id"],
                        count=self.settings.hard_negatives_per_query,
                    )

            except Exception as e:
                logger.error(f"Error processing chunk {chunk['id']}: {e}")
                skipped += 1
                continue

        # Export datasets
        print("\nWriting output files...")
        self.exporter.export_qa_pairs(qa_pairs, chunks)
        self.exporter.export_retrieval_eval(qa_pairs, hard_negatives)
        self.exporter.export_rag_eval(qa_pairs)

        # Print summary
        self.exporter.print_summary(
            total_chunks=len(chunks),
            total_qa_pairs=len(qa_pairs),
            skipped_chunks=skipped,
        )

        return {
            "chunks": len(chunks),
            "qa_pairs": len(qa_pairs),
            "skipped": skipped,
            "qa_pairs_data": qa_pairs,
            "chunks_data": chunks,
            "hard_negatives": hard_negatives,
        }

    def load_to_catalog(
        self,
        qa_pairs: list[dict[str, Any]],
        version: str,
        catalog_name: str = "verdict",
    ) -> int:
        """
        Load generated Q&A pairs into Unity Catalog.

        Args:
            qa_pairs: List of Q&A pair dicts.
            version: Dataset version string.
            catalog_name: Unity Catalog name.

        Returns:
            Number of prompts loaded.
        """
        try:
            from verdict.data.prompt_dataset import PromptDatasetManager
        except ImportError:
            logger.error(
                "Unity Catalog integration requires pyspark. "
                "Install with: pip install pyspark"
            )
            return 0

        manager = PromptDatasetManager(catalog_name=catalog_name)

        # Transform Q&A pairs to prompt format
        prompts = [
            {
                "prompt": pair["question"],
                "ground_truth": pair["ground_truth"],
                "metadata": {
                    "chunk_id": pair["chunk_id"],
                    "source": "verdict-testgen",
                },
            }
            for pair in qa_pairs
        ]

        count = manager.create_dataset(prompts=prompts, version=version)
        logger.info(f"Loaded {count} prompts into catalog as version '{version}'")
        return count


def main() -> None:
    """CLI for RAG test dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate RAG test datasets from Qdrant collection"
    )
    parser.add_argument(
        "--collection",
        help="Override QDRANT_COLLECTION from env",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Override QDRANT_SCROLL_LIMIT from env",
    )
    parser.add_argument(
        "--output-dir",
        help="Override OUTPUT_DIR from env",
    )
    parser.add_argument(
        "--filter-field",
        help="Qdrant payload field to filter by",
    )
    parser.add_argument(
        "--filter-value",
        help="Value to match for filter field",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch chunks and preview without calling LLM",
    )
    parser.add_argument(
        "--load-to-catalog",
        action="store_true",
        help="Load QA pairs into Unity Catalog raw.prompt_datasets",
    )
    parser.add_argument(
        "--dataset-version",
        help="Dataset version string (required with --load-to-catalog)",
    )
    parser.add_argument(
        "--catalog",
        default="verdict",
        help="Catalog name (default: verdict)",
    )

    args = parser.parse_args()

    # Load settings with overrides
    settings_kwargs: dict[str, Any] = {}
    if args.collection:
        settings_kwargs["qdrant_collection"] = args.collection
    if args.limit:
        settings_kwargs["qdrant_scroll_limit"] = args.limit
    if args.output_dir:
        settings_kwargs["output_dir"] = args.output_dir

    try:
        settings = Settings(**settings_kwargs)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error(
            "Ensure required environment variables are set: "
            "QDRANT_COLLECTION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
        )
        sys.exit(1)

    # Validate catalog loading requirements
    if args.load_to_catalog and not args.dataset_version:
        logger.error("--dataset-version is required when using --load-to-catalog")
        sys.exit(1)

    # Create generator and run
    generator = TestDatasetGenerator(settings)

    result = generator.generate(
        filter_field=args.filter_field,
        filter_value=args.filter_value,
        dry_run=args.dry_run,
    )

    # Optionally load to catalog
    if args.load_to_catalog and result.get("qa_pairs", 0) > 0:
        qa_pairs_data = result.get("qa_pairs_data", [])
        generator.load_to_catalog(
            qa_pairs=qa_pairs_data,
            version=args.dataset_version,
            catalog_name=args.catalog,
        )


if __name__ == "__main__":
    main()
