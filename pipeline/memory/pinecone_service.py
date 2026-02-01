"""Vector database integration for long-term memory.

Utilizes Pinecone as a backend for semantic search and Sentence Transformers 
for producing embeddings. Supports upserting, querying, and managing memories
with metadata filtering.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from pipeline.config import PipelineConfig
from pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from logging import Logger

    from pinecone import Index, Pinecone
    from sentence_transformers import SentenceTransformer


class MemoryEntry(TypedDict):
    """Standardized representation of a retrieved memory record.

    Attributes:
        id: Unique identifier for the memory in the vector index.
        content: The raw text content or preview associated with the memory.
        metadata: Key-value pairs containing source info, timestamps, and custom data.
        score: Semantic similarity score (if retrieved via search).
    """
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None


class PineconeMemory:
    """Memory management service powered by Pinecone.

    Handles the lifecycle of semantic memories, including automatic index 
    provisioning, batch upserts, and vector-based retrieval with metadata 
    constraints.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config: PipelineConfig = config
        self.logger: Logger = get_logger(__name__)
        self._client: Pinecone | None = None
        self._index: Index | None = None
        self._model: SentenceTransformer | None = None
        
        self.enabled: bool = PINECONE_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
        if not self.enabled:
            self.logger.warning("Pinecone or Sentence Transformers not available. Memory disabled.")
            return

        self._api_key = config.memory.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self._index_name = config.memory.pinecone_index_name or os.environ.get("PINECONE_INDEX_NAME", "browser-use-memory")
        
        if not self._api_key:
            self.logger.warning("No Pinecone API key found. Memory disabled.")
            self.enabled = False

    def _initialize(self) -> None:
        """Lazy initialization of Pinecone client and model."""
        if not self.enabled or self._client:
            return

        try:
            self.logger.info("Initializing Pinecone client...")
            self._client = Pinecone(api_key=self._api_key)
            
            # Check if index exists, if not create it (Serverless)
            existing_indexes = [i.name for i in self._client.list_indexes()]
            if self._index_name not in existing_indexes:
                self.logger.info(f"Creating Pinecone index: {self._index_name}")
                self._client.create_index(
                    name=self._index_name,
                    dimension=384,  # Standard for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )

            self._index = self._client.Index(self._index_name)
            
            self.logger.info("Loading embedding model...")
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory service: {e}")
            self.enabled = False

    def upsert(self, content: str, source_file: str, metadata: dict[str, Any] | None = None) -> bool:
        """Store content in memory."""
        self._initialize()
        if not self.enabled or not self._index:
            return False

        try:
            vector = self._model.encode(content).tolist()
            
            doc_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            meta = {
                "source": source_file,
                "created_at": now,
                "text": content[:1000]  # Store first 1000 chars as metadata text preview
            }
            if metadata:
                meta.update(metadata)

            self._index.upsert(vectors=[(doc_id, vector, meta)])
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert memory: {e}")
            return False

    def query(self, query_text: str, top_k: int = 5, filter_dict: dict[str, Any] | None = None) -> list[MemoryEntry]:
        """Retrieve relevant memories with optional metadata filtering."""
        self._initialize()
        if not self.enabled or not self._index:
            return []

        try:
            vector = self._model.encode(query_text).tolist()
            
            results = self._index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            memories = []
            for match in results.matches:
                memories.append({
                    "id": match.id,
                    "content": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                    "score": match.score
                })
                
            return memories

        except Exception as e:
            self.logger.error(f"Failed to query memory: {e}")
            return []

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        self._initialize()
        if not self.enabled or not self._index:
            return False

        try:
            self._index.delete(ids=[memory_id])
            self.logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory: {e}")
            return False

    def delete_by_filter(self, filter_dict: dict[str, Any]) -> bool:
        """Delete memories matching a filter."""
        self._initialize()
        if not self.enabled or not self._index:
            return False

        try:
            self._index.delete(filter=filter_dict)
            self.logger.info(f"Deleted memories with filter: {filter_dict}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete by filter: {e}")
            return False

    def update(self, memory_id: str, content: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
        """Update an existing memory's content and/or metadata."""
        self._initialize()
        if not self.enabled or not self._index:
            return False

        try:
            # Fetch existing record
            fetch_result = self._index.fetch(ids=[memory_id])
            if memory_id not in fetch_result.vectors:
                self.logger.error(f"Memory {memory_id} not found")
                return False

            existing = fetch_result.vectors[memory_id]
            
            # Prepare updated data
            new_vector = existing.values
            new_metadata = existing.metadata.copy() if existing.metadata else {}
            
            if content is not None:
                new_vector = self._model.encode(content).tolist()
                new_metadata["text"] = content[:1000]
                new_metadata["updated_at"] = datetime.now().isoformat()
            
            if metadata:
                new_metadata.update(metadata)

            self._index.upsert(vectors=[(memory_id, new_vector, new_metadata)])
            self.logger.info(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update memory: {e}")
            return False

    def batch_upsert(self, items: list[dict[str, Any]]) -> bool:
        """Batch upsert multiple memories for efficiency.
        
        Args:
            items: List of dictionaries with keys: 'content', 'source', optional 'id', 'metadata'
        """
        self._initialize()
        if not self.enabled or not self._index:
            return False

        try:
            vectors = []
            now = datetime.now().isoformat()
            
            for item in items:
                content = item["content"]
                vector = self._model.encode(content).tolist()
                doc_id = item.get("id", str(uuid.uuid4()))
                
                meta = {
                    "source": item.get("source", "unknown"),
                    "created_at": now,
                    "text": content[:1000]
                }
                if "metadata" in item:
                    meta.update(item["metadata"])
                
                vectors.append((doc_id, vector, meta))
            
            # Pinecone recommends batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self._index.upsert(vectors=batch)
            
            self.logger.info(f"Batch upserted {len(vectors)} memories")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to batch upsert: {e}")
            return False

    def clear_all(self) -> bool:
        """Delete all memories in the index. Use with caution!"""
        self._initialize()
        if not self.enabled or not self._index:
            return False

        try:
            self._index.delete(delete_all=True)
            self.logger.warning("Cleared all memories from index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear memories: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics including vector count and dimension."""
        self._initialize()
        if not self.enabled or not self._index:
            return {}

        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

    def fetch(self, memory_ids: list[str]) -> list[MemoryEntry]:
        """Fetch specific memories by their IDs without semantic search."""
        self._initialize()
        if not self.enabled or not self._index:
            return []

        try:
            result = self._index.fetch(ids=memory_ids)
            
            memories = []
            for mem_id, vector_data in result.vectors.items():
                memories.append({
                    "id": mem_id,
                    "content": vector_data.metadata.get("text", ""),
                    "metadata": vector_data.metadata,
                    "score": None
                })
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to fetch memories: {e}")
            return []

    def search_by_metadata(self, filter_dict: dict[str, Any], top_k: int = 10) -> list[MemoryEntry]:
        """Search memories by metadata filter without semantic search.
        
        Note: This uses a dummy vector since Pinecone requires a query vector.
        For pure metadata filtering, consider using fetch with list_paginated.
        """
        self._initialize()
        if not self.enabled or not self._index:
            return []

        try:
            # Use a zero vector to essentially do metadata-only filtering
            # This is a workaround since Pinecone requires a query vector
            dummy_vector = [0.0] * 384
            
            results = self._index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            memories = []
            for match in results.matches:
                memories.append({
                    "id": match.id,
                    "content": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                    "score": None  # Score not meaningful for metadata-only search
                })
                
            return memories

        except Exception as e:
            self.logger.error(f"Failed to search by metadata: {e}")
            return []

    def list_by_source(self, source_file: str, top_k: int = 100) -> list[MemoryEntry]:
        """List all memories from a specific source file."""
        return self.search_by_metadata(filter_dict={"source": {"$eq": source_file}}, top_k=top_k)

    def get_recent(self, limit: int = 10, since: str | None = None) -> list[MemoryEntry]:
        """Get recent memories, optionally since a specific datetime."""
        filter_dict = None
        if since:
            filter_dict = {"created_at": {"$gte": since}}
        
        return self.search_by_metadata(filter_dict=filter_dict or {}, top_k=limit)

    def close(self) -> None:
        """Clean up resources."""
        self._client = None
        self._index = None
        self._model = None
        self.logger.info("Memory service closed")