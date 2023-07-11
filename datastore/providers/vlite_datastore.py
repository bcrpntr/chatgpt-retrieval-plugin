import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger

from vlite import VLite
from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    DocumentChunkWithScore,
)

class VLiteDataStore(DataStore):
    def __init__(self):
        self.db = VLite()  # Initialize VLite

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a dict of document_ids to list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
        for document_id, document_chunks in chunks.items():
            for chunk in document_chunks:
                # Memorize chunk using VLite
                self.db.memorize(chunk.text, id=document_id, metadata=chunk.metadata)

        return list(chunks.keys())

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        query_results: List[QueryResult] = []

        for query in queries:
            # Remember vectors using VLite
            results, scores = self.db.remember(text=query.query, top_k=query.top_k)

            # Convert results to QueryResult format
            query_results.append(self._convert_results_to_query_result_format(results, scores))

        return query_results

    def _convert_results_to_query_result_format(self, results, scores):
        """
        Convert results to QueryResult format.
        """
        document_chunks_with_score = [
            DocumentChunkWithScore(
                id=None,  # We don't have an id in this case
                text=result,
                score=float(score),
                metadata=None,  # We don't have metadata in this case
            )
            for result, score in zip(results, scores)
        ]

        return QueryResult(query=None, results=document_chunks_with_score)
