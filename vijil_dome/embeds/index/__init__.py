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

from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from vijil_dome.embeds import EmbeddingsItem


class AbstractEmbeddingsIndex(ABC):
    @abstractmethod
    def get_embedding_size(self) -> int:
        """get the size of the embeddings."""
        pass

    @abstractmethod
    async def add_item(self, item: EmbeddingsItem) -> None:
        """Add a single item to the index."""
        pass

    @abstractmethod
    async def add_items(self, items: List[EmbeddingsItem]) -> None:
        """Add multiple items to the index."""
        pass

    @abstractmethod
    async def build(self) -> None:
        """Build the index."""
        pass

    @abstractmethod
    async def persist(self, path: str) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    async def nearest_neighbor(
        self, query: str, k: int, with_distance: bool = False
    ) -> List[Tuple[EmbeddingsItem, Optional[float]]]:
        """
        Perform approximate nearest neighbor search.
        The search should return the k nearest neighbors to the query.
        If with_distance is True, the search should return the distance to each neighbor.
        Otherwise, the search should return only the neighbors and distances should be None.
        """
        pass
