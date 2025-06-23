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
import pandas as pd
from abc import abstractmethod
from scipy import spatial
from typing import List, Tuple, Optional, Union
from vijil_dome.embeds import EmbeddingsItem
from vijil_dome.embeds.index import AbstractEmbeddingsIndex
from vijil_dome.embeds import AbstractEmbedder
from vijil_dome.types import Embeddings

try:
    from annoy import AnnoyIndex

    _ANNOY_AVAILABLE = True
except ImportError:
    logging.warning(
        "Annoy was not installed. Only in-memory indexing will be available."
    )
    _ANNOY_AVAILABLE = False


class AbstractBaseEmbeddingsIndex(AbstractEmbeddingsIndex):
    _index: Optional[Union[pd.DataFrame, "AnnoyIndex"]] = None
    _embedder: Optional[AbstractEmbedder] = None
    _items: List[EmbeddingsItem] = []
    _embeddings: List[Embeddings] = []
    _embedding_size: int = 0

    @abstractmethod
    async def build(self) -> None:
        pass

    async def persist(self, path: str) -> None:
        pass

    async def add_item(self, item: EmbeddingsItem) -> None:
        await self.add_items([item])

    async def add_items(self, items: List[EmbeddingsItem]) -> None:
        """Add multiple items to the index.

        Args:
            items (List[EmbeddingsItem]): The list of items to add to the index.

        """
        if self._embedder is None:
            raise ValueError("Embedder not available")

        if self._index is None:  # skip if index is already built
            # Compute embeddings only for items that don't have them
            new_items = [item for item in items if item.embeddings is not None]
            new_embeddings = [
                item.embeddings for item in items if item.embeddings is not None
            ]

            missing_items = [item for item in items if item.embeddings is None]
            missing_embeddings = []
            if len(missing_items) > 0:
                texts_to_compute_embeddings = [item.text for item in missing_items]
                missing_embeddings.extend(
                    await self._embedder.embeddings(texts_to_compute_embeddings)
                )

            new_items.extend(missing_items)
            new_embeddings.extend(missing_embeddings)
            self._embeddings.extend(new_embeddings)

            logging.debug(f"\n\nAdded {len(new_items)} embeddings to be indexed")

        self._items.extend(new_items)

        # Update the embedding if it was not computed up to this point
        self._embedding_size = len(self._embeddings[0])

        logging.debug(f"\n\nAdded {len(new_items)} items")
        logging.debug(f"Current # of items: {len(self._items)}")
        logging.debug(f"Embedding size: {self._embedding_size}")
        logging.debug(f"Items: {self._items}\n\n")


class InMemEmbeddingsIndex(AbstractBaseEmbeddingsIndex):
    def __init__(self, embedder):
        self._embedder = embedder

    async def build(self) -> None:
        """Build the DF index."""
        self._index: pd.DataFrame = pd.DataFrame(
            index=np.arange(0, len(self._embeddings)), columns=["embedding"]
        )
        for i in range(len(self._embeddings)):
            self._index.loc[i] = [self._embeddings[i]]

    async def nearest_neighbor(
        self, query: str, k: int, with_distance: bool = False
    ) -> List[Tuple[EmbeddingsItem, Optional[float]]]:
        """Perform approximate nearest neighbor search.

        Args:
            query (str): The query text.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[EmbeddingsItem]: The list of nearest neighbors.

        """
        if self._embedder is None:
            raise ValueError(
                "Embedder not available. Please provide an embedder before performing ANN."
            )
        if self._index is None:
            raise ValueError(
                "Index not built. Please build the index before searching for Nearest Neighbor."
            )
        _embedding = (await self._embedder.embeddings([query]))[0]
        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
        index_and_relatednesses = [
            (i, relatedness_fn(_embedding, row["embedding"]))
            for i, row in self._index.iterrows()
        ]
        index_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        if len(index_and_relatednesses) > k:
            index_and_relatednesses = index_and_relatednesses[:k]
        if with_distance:
            return [(self._items[i[0]], i[1]) for i in index_and_relatednesses]
        else:
            return [(self._items[i[0]], None) for i in index_and_relatednesses]

    def get_embedding_size(self) -> int:
        return self._embedding_size


class AnnoyEmbeddingsIndex(AbstractBaseEmbeddingsIndex):
    """
    This class represents an embeddings index that uses the Annoy library for efficient nearest neighbor search.
    It supports batch processing of embeddings and provides methods for adding items to the index and building the index.

    Attributes:
        embedding_model (Optional[str]): The name of the embedding model to use.
        embedding_engine (Optional[str]): The name of the embedding engine to use.
        _model (Optional[AbstractEmbeddingsModel]): The model used for computing the embeddings.
        _items (List[EmbeddingsItem]): The list of items in the index.
        _embeddings (List[EmbeddingsItem]): The list of embeddings in the index.
        _embedding_size (int): The size of the embeddings.

    """

    def __init__(self, embedder):
        if not _ANNOY_AVAILABLE:
            raise ValueError(
                "Annoy is not installed. Only in-memory indexing is available. Use `pip install annoy` to intall it."
            )
        self._embedder = embedder

    async def build(self) -> None:
        """Build the Annoy index."""
        self._index = AnnoyIndex(len(self._embeddings[0]), "angular")
        for i in range(len(self._embeddings)):
            self._index.add_item(i, self._embeddings[i])
        self._index.build(10)

    async def nearest_neighbor(
        self, query: str, k: int, with_distance: bool = False
    ) -> List[Tuple[EmbeddingsItem, Optional[float]]]:
        """Perform approximate nearest neighbor search.

        Args:
            query (str): The query text.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[EmbeddingsItem]: The list of nearest neighbors.

        """
        if self._embedder is None:
            raise ValueError("Embedder not available.")
        if self._index is None:
            raise ValueError(
                "Index not built. Please build the index before performing nearest neighbor search."
            )
        embeddings = await self._embedder.embeddings([query])
        query_embedding = embeddings[0]
        if with_distance:
            ind_distances = self._index.get_nns_by_vector(
                query_embedding, k, include_distances=True
            )
            with_distance_list = zip(
                [self._items[i] for i in ind_distances[0]], ind_distances[1]
            )
            return list(with_distance_list)
        else:
            indices = self._index.get_nns_by_vector(query_embedding, k)
            return [(self._items[i], None) for i in indices]

    def get_embedding_size(self) -> int:
        return self._embedding_size
