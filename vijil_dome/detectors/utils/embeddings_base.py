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

import asyncio
import nest_asyncio
from typing import Union  # noqa: F401
from vijil_dome.embeds import EmbeddingsItem
from vijil_dome.embeds.embedder import Embedder
from vijil_dome.embeds.index.basic import InMemEmbeddingsIndex, AnnoyEmbeddingsIndex
from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.utils.readers import normalize_string, read_lines


class BaseEmbeddingsDetector(DetectionMethod):
    def __init__(
        self,
        method_name: str,
        data_filepath: str,
        engine: str,
        model: str,
        threshold: float,
        in_mem: bool,
    ):
        self.data_file = data_filepath
        self.in_mem = in_mem
        self.threshold = threshold
        self.embeddings_engine = engine
        self.embeddings_model = model
        self.blocked_response_string = f"Method:{method_name}"

        self.embedder = Embedder(
            embedding_engine=self.embeddings_engine,
            embedding_model=self.embeddings_model,
        )

        if self.in_mem:
            self.embedding_index = InMemEmbeddingsIndex(embedder=self.embedder)  # type: Union[InMemEmbeddingsIndex, AnnoyEmbeddingsIndex]
        else:
            self.embedding_index = AnnoyEmbeddingsIndex(embedder=self.embedder)

        self.index_items = []
        lines_iter = read_lines(self.data_file)
        for id, line in enumerate(lines_iter):
            line_lower = line.lower()
            normed = normalize_string(line_lower)
            self.index_items.append(EmbeddingsItem(text=normed, meta={"id": str(id)}))

        # This is the only method that has an async in the init
        # Need to nest the asyncio calls here so you can init it if its used inside the Async OAI client
        nest_asyncio.apply()
        asyncio.run(self.embedding_index.add_items(self.index_items))
        # Build the index
        asyncio.run(self.embedding_index.build())

    async def detect(self, query_string: str) -> DetectionResult:
        res = await self.embedding_index.nearest_neighbor(
            normalize_string(query_string.lower()), k=1, with_distance=True
        )
        sim = res[0][1]
        if sim is None:
            raise ValueError("Embeddings search returned None distance")
        flagged = sim > self.threshold
        return flagged, {
            "type": type(self),
            "search_results": res,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
