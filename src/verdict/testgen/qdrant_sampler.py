"""
Qdrant sampler for fetching document chunks from a collection.

Samples chunks from a Qdrant collection for RAG test data generation.
"""

import logging
import random
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum text length for a valid chunk
MIN_CHUNK_LENGTH = 50


class QdrantSampler:
    """
    Fetches and samples document chunks from a Qdrant collection.

    Args:
        url: Qdrant connection URL.
        api_key: Optional Qdrant Cloud API key.
        collection: Collection name to sample from.
        text_field: Payload field containing chunk text.
        scroll_limit: Maximum number of chunks to fetch.
    """

    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        text_field: str = "text",
        scroll_limit: int = 200,
    ) -> None:
        self.url = url
        self.collection = collection
        self.text_field = text_field
        self.scroll_limit = scroll_limit

        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Connected to Qdrant at {url}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Qdrant at {url}: {e}"
            ) from e

    def fetch_chunks(
        self,
        filter_field: str | None = None,
        filter_value: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch chunks from the Qdrant collection.

        Args:
            filter_field: Optional payload field to filter by.
            filter_value: Value to match for the filter field.

        Returns:
            List of dicts with keys: id, text, metadata.

        Raises:
            ValueError: If collection does not exist.
        """
        chunks: list[dict[str, Any]] = []
        offset: str | None = None
        page_size = 100

        # Build filter if provided
        query_filter = None
        if filter_field and filter_value:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key=filter_field,
                        match=MatchValue(value=filter_value),
                    )
                ]
            )

        try:
            # Verify collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection not in collection_names:
                raise ValueError(
                    f"Collection '{self.collection}' not found. "
                    f"Available: {collection_names}"
                )

            # Scroll through collection in pages
            while len(chunks) < self.scroll_limit:
                results, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=min(page_size, self.scroll_limit - len(chunks)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=query_filter,
                )

                for point in results:
                    if point.payload is None:
                        continue

                    text = point.payload.get(self.text_field, "")
                    if not text or len(str(text).strip()) < MIN_CHUNK_LENGTH:
                        logger.warning(
                            f"Skipping chunk {point.id}: text too short or empty"
                        )
                        continue

                    # Extract metadata from payload (excluding text field)
                    metadata = {
                        k: v for k, v in point.payload.items()
                        if k != self.text_field
                    }

                    chunks.append({
                        "id": str(point.id),
                        "text": str(text).strip(),
                        "metadata": metadata,
                    })

                if offset is None:
                    break

            logger.info(f"Fetched {len(chunks)} chunks from '{self.collection}'")
            return chunks[: self.scroll_limit]

        except UnexpectedResponse as e:
            raise RuntimeError(
                f"Qdrant API error for collection '{self.collection}': {e}"
            ) from e

    def get_hard_negatives(
        self,
        chunks: list[dict[str, Any]],
        source_id: str,
        count: int = 2,
    ) -> list[str]:
        """
        Get hard negative chunk IDs for a query.

        Selects random chunks that are not the source chunk, preferring
        chunks from different sources if metadata is available.

        Args:
            chunks: List of all fetched chunks.
            source_id: ID of the source chunk to exclude.
            count: Number of hard negatives to return.

        Returns:
            List of chunk IDs for hard negatives.
        """
        # Get source chunk's metadata
        source_chunk = next(
            (c for c in chunks if c["id"] == source_id), None
        )
        source_doc = (
            source_chunk.get("metadata", {}).get("source")
            if source_chunk else None
        )

        # Prefer chunks from different documents
        candidates = [
            c for c in chunks
            if c["id"] != source_id
        ]

        if source_doc:
            different_source = [
                c for c in candidates
                if c.get("metadata", {}).get("source") != source_doc
            ]
            same_source = [
                c for c in candidates
                if c.get("metadata", {}).get("source") == source_doc
            ]
            # Prioritize different sources
            candidates = different_source + same_source

        if len(candidates) <= count:
            return [c["id"] for c in candidates]

        return [c["id"] for c in random.sample(candidates, count)]
