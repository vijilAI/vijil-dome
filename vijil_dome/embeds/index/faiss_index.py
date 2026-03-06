# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# vijil and vijil-dome are trademarks owned by Vijil Inc.

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from vijil_dome.embeds import EmbeddingsItem
from vijil_dome.embeds.index import AbstractEmbeddingsIndex
from vijil_dome.embeds import AbstractEmbedder
from vijil_dome.types import Embeddings

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

logger = logging.getLogger("vijil.dome")


class FaissEmbeddingsIndex(AbstractEmbeddingsIndex):
    """
    FAISS-based embeddings index for efficient similarity search.
    
    Supports loading pre-built FAISS indices from disk or S3.
    The index should be built externally and loaded here for querying.
    """
    
    def __init__(self, embedder: Optional[AbstractEmbedder] = None):
        """Initialize FAISS embeddings index.
        
        Args:
            embedder: Embedder for encoding queries. Required for nearest_neighbor search.
        """
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is not installed. Install it with: pip install faiss-cpu"
            )
        self._embedder = embedder
        self._index: Optional["faiss.Index"] = None
        self._items: List[EmbeddingsItem] = []
        self._embedding_size: int = 0
        self._is_loaded = False

    def get_embedding_size(self) -> int:
        """Get the size of the embeddings."""
        if self._index is None:
            raise ValueError("Index not loaded. Call load_from_file() first.")
        return self._embedding_size

    async def add_item(self, item: EmbeddingsItem) -> None:
        """Add a single item to the index.
        
        Note: This is typically not used for pre-built FAISS indices.
        Use load_from_file() instead.
        """
        await self.add_items([item])

    async def add_items(self, items: List[EmbeddingsItem]) -> None:
        """Add multiple items to the index.
        
        Note: This is typically not used for pre-built FAISS indices.
        Use load_from_file() instead.
        """
        if self._is_loaded:
            raise ValueError(
                "Cannot add items to a loaded FAISS index. "
                "FAISS indices should be built externally and loaded here."
            )
        # For compatibility, but building should be done externally
        self._items.extend(items)

    async def build(self) -> None:
        """Build the index.
        
        Note: For FAISS, indices are typically built externally.
        This method is provided for compatibility but will raise an error
        if called without a pre-built index.
        """
        if self._index is None:
            raise ValueError(
                "FAISS index must be loaded from file. "
                "Use load_from_file() to load a pre-built index."
            )

    async def persist(self, path: str) -> None:
        """Persist the index to disk."""
        if self._index is None:
            raise ValueError("Index not loaded. Cannot persist.")
        faiss.write_index(self._index, path)
        logger.info(f"Persisted FAISS index to {path}")

    def load_from_file(self, file_path: str) -> None:
        """Load a pre-built FAISS index from disk.
        
        Args:
            file_path: Path to the FAISS index file (.index)
            
        Raises:
            FileNotFoundError: If the index file doesn't exist
            ValueError: If the index cannot be loaded
        """
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is not installed. Install it with: pip install faiss-cpu"
            )
        
        index_path = Path(file_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {file_path}")
        
        try:
            self._index = faiss.read_index(str(index_path))
            self._embedding_size = self._index.d
            self._is_loaded = True
            logger.info(f"Loaded FAISS index from {file_path} (dimension: {self._embedding_size})")
        except Exception as e:
            raise ValueError(f"Failed to load FAISS index from {file_path}: {e}")

    async def nearest_neighbor(
        self, query: str, k: int, with_distance: bool = False
    ) -> List[Tuple[EmbeddingsItem, Optional[float]]]:
        """Perform approximate nearest neighbor search using FAISS.
        
        Args:
            query: The query text to search for
            k: The number of nearest neighbors to return
            with_distance: If True, return similarity distances
            
        Returns:
            List of tuples containing (EmbeddingsItem, distance) pairs.
            Distance is None if with_distance=False.
            
        Raises:
            ValueError: If index not loaded or embedder not available
        """
        if self._index is None:
            raise ValueError("Index not loaded. Call load_from_file() first.")
        if self._embedder is None:
            raise ValueError(
                "Embedder not available. Provide an embedder to encode queries."
            )
        if len(self._items) == 0:
            raise ValueError(
                "Items not set. Call set_items() to provide mapping between FAISS indices and EmbeddingsItem objects."
            )
        
        # Encode query
        query_embeddings = await self._embedder.embeddings([query])
        query_vector = np.array([query_embeddings[0]], dtype=np.float32)
        
        # Ensure query vector matches index dimension
        if query_vector.shape[1] != self._embedding_size:
            error_msg = (
                f"Query embedding dimension ({query_vector.shape[1]}) "
                f"does not match index dimension ({self._embedding_size}). "
            )
            # Provide helpful suggestions based on common dimensions
            if self._embedding_size == 1536:
                error_msg += (
                    "The index was likely built with OpenAI's text-embedding-ada-002. "
                    "Please use a matching embedding model or rebuild the index."
                )
            elif self._embedding_size == 384:
                error_msg += (
                    "The index was likely built with all-MiniLM-L6-v2. "
                    "Please ensure your embedding_model matches."
                )
            elif self._embedding_size == 768:
                error_msg += (
                    "The index was likely built with all-mpnet-base-v2 or similar. "
                    "Please ensure your embedding_model matches."
                )
            else:
                error_msg += (
                    "Please ensure your embedding_model produces embeddings "
                    f"with dimension {self._embedding_size}."
                )
            raise ValueError(error_msg)
        
        # Search FAISS index
        if with_distance:
            distances, indices = self._index.search(query_vector, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self._items):
                    # FAISS returns squared L2 distances, convert to similarity
                    # Using 1 / (1 + distance) as similarity score
                    distance = float(distances[0][i])
                    similarity = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
                    results.append((self._items[idx], similarity))
                else:
                    logger.warning(f"Index {idx} out of range for {len(self._items)} items")
            return results
        else:
            distances, indices = self._index.search(query_vector, k)
            results = []
            for idx in indices[0]:
                if idx < len(self._items):
                    results.append((self._items[idx], None))
                else:
                    logger.warning(f"Index {idx} out of range for {len(self._items)} items")
            return results

    def set_items(self, items: List[EmbeddingsItem]) -> None:
        """Set the items list for mapping FAISS indices to EmbeddingsItem objects.
        
        This should be called after loading the index to provide the mapping
        between FAISS index positions and actual items.
        
        Args:
            items: List of EmbeddingsItem objects, ordered to match FAISS index positions
            
        Raises:
            ValueError: If index not loaded or items count doesn't match index size
        """
        if self._index is None:
            raise ValueError("Index must be loaded before setting items. Call load_from_file() first.")
        
        if len(items) != self._index.ntotal:
            raise ValueError(
                f"Items count ({len(items)}) does not match FAISS index size ({self._index.ntotal}). "
                f"Ensure items are ordered to match the index positions."
            )
        
        self._items = items
        logger.info(f"Set {len(items)} items for FAISS index mapping")
