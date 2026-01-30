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
    """Structure of a memory entry."""
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None


class PineconeMemory:
    """Memory service using Pinecone vector database."""

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

    def query(self, query_text: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve relevant memories."""
        self._initialize()
        if not self.enabled or not self._index:
            return []

        try:
            vector = self._model.encode(query_text).tolist()
            
            results = self._index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
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
