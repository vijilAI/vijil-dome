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

from typing import List, Optional
from abc import ABC, abstractmethod

from vijil_dome.types import Sentences

# Optional executor for asynchronous tasks, defaulting to asyncio's native executor.
embeddings_executor = None

# Cache for instantiated embedding models to ensure they are used as singletons.
_embedding_model_cache = {}


class AbstractEmbeddingsModel(ABC):
    """Abstract base class for embedding models.

    This class defines a generic interface for embedding models, which are responsible for
    converting input texts into their corresponding embedding vectors."""

    @abstractmethod
    async def encode_async(self, sentences: Sentences) -> List[List[float]]:
        """Asynchronously encode documents into embeddings.

        Args:
            sentences (Sentences): The sentences to be encoded.

        Returns:
            List[List[float]]: A list containing the embedding vectors for each document.
        """
        pass


def init_embedding_model(
    embedding_model: str, embedding_engine: str
) -> AbstractEmbeddingsModel:
    """Factory function to initialize and cache embedding models.

    Args:
        embedding_model (str): Identifier for the embedding model.
        embedding_engine (str): The embedding engine to use.

    Returns:
        EmbeddingModel: An instance of the specified embedding model.

    Raises:
        ValueError: If the specified embedding engine is not supported.
    """
    model_key = f"{embedding_engine}-{embedding_model}"

    model: Optional[AbstractEmbeddingsModel] = None
    if model_key not in _embedding_model_cache:
        if embedding_engine == "SentenceTransformers":
            from .sentence_transformers import SentenceTransformerEmbeddingModel

            model = SentenceTransformerEmbeddingModel(embedding_model)
        elif embedding_engine == "FastEmbed":
            from .fastembed import FastEmbedEmbeddingModel

            model = FastEmbedEmbeddingModel(embedding_model)
        # elif embedding_engine == "openai":
        #     from .openai import OpenAIEmbeddingModel
        #     model = OpenAIEmbeddingModel(embedding_model)
        else:
            raise ValueError(f"Invalid embedding engine: {embedding_engine}")

        _embedding_model_cache[model_key] = model

    return _embedding_model_cache[model_key]
