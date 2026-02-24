"""
Exporters for RAG test datasets.

Writes output files in RAGAS-compatible formats.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Exports generated datasets to various formats.

    Args:
        output_dir: Directory to write output files.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def export_qa_pairs(
        self,
        qa_pairs: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> str:
        """
        Export Q&A pairs to JSONL format.

        Args:
            qa_pairs: List of Q&A pair dicts.
            chunks: List of source chunks (for metadata).

        Returns:
            Path to the output file.
        """
        output_path = self.output_dir / "qa_pairs.jsonl"

        # Build metadata lookup
        chunk_metadata = {c["id"]: c.get("metadata", {}) for c in chunks}

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in qa_pairs:
                record = {
                    "chunk_id": pair["chunk_id"],
                    "question": pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "source_text": pair["source_text"],
                    "metadata": chunk_metadata.get(pair["chunk_id"], {}),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(qa_pairs)} Q&A pairs to {output_path}")
        return str(output_path)

    def export_retrieval_eval(
        self,
        qa_pairs: list[dict[str, Any]],
        hard_negatives: dict[str, list[str]],
    ) -> str:
        """
        Export retrieval evaluation dataset to JSONL format.

        Args:
            qa_pairs: List of Q&A pair dicts.
            hard_negatives: Dict mapping chunk_id to list of hard negative IDs.

        Returns:
            Path to the output file.
        """
        output_path = self.output_dir / "retrieval_eval.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in qa_pairs:
                record = {
                    "query": pair["question"],
                    "relevant_chunk_ids": [pair["chunk_id"]],
                    "hard_negative_chunk_ids": hard_negatives.get(
                        pair["chunk_id"], []
                    ),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(qa_pairs)} retrieval records to {output_path}")
        return str(output_path)

    def export_rag_eval(
        self,
        qa_pairs: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Export RAG evaluation dataset in JSONL and HuggingFace formats.

        Args:
            qa_pairs: List of Q&A pair dicts.

        Returns:
            Tuple of (jsonl_path, hf_dataset_path).
        """
        # Export JSONL
        jsonl_path = self.output_dir / "rag_eval.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for pair in qa_pairs:
                record = {
                    "question": pair["question"],
                    "contexts": [pair["source_text"]],
                    "ground_truth": pair["ground_truth"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(qa_pairs)} RAG eval records to {jsonl_path}")

        # Export HuggingFace Dataset
        hf_path = self.output_dir / "rag_eval_hf"
        records = [
            {
                "question": pair["question"],
                "contexts": [pair["source_text"]],
                "ground_truth": pair["ground_truth"],
            }
            for pair in qa_pairs
        ]

        dataset = Dataset.from_list(records)
        dataset.save_to_disk(str(hf_path))

        logger.info(f"Exported HuggingFace Dataset to {hf_path}")

        return str(jsonl_path), str(hf_path)

    def print_summary(
        self,
        total_chunks: int,
        total_qa_pairs: int,
        skipped_chunks: int,
    ) -> None:
        """
        Print a summary table of the export.

        Args:
            total_chunks: Total chunks processed.
            total_qa_pairs: Total Q&A pairs generated.
            skipped_chunks: Number of skipped chunks.
        """
        print("\n" + "=" * 60)
        print("RAG Test Dataset Generation Summary")
        print("=" * 60)
        print(f"  Chunks processed:    {total_chunks}")
        print(f"  Q&A pairs generated: {total_qa_pairs}")
        print(f"  Chunks skipped:      {skipped_chunks}")
        print("\nOutput files:")
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f.name:<20} ({size:,} bytes)")
            elif f.is_dir():
                print(f"  {f.name:<20} (HuggingFace Dataset)")
        print("=" * 60 + "\n")
