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
from typing import List
from vijil_dome.types import Sentences
from vijil_dome.embeds.models import AbstractEmbeddingsModel


class OpenAIEmbeddingModel(AbstractEmbeddingsModel):
    """Embedding model using OpenAI's API.
    
    This class represents an embedding model that utilizes OpenAI's embedding API
    for generating sentence embeddings.
    
    Args:
        embedding_model (str): The OpenAI embedding model name (e.g., "text-embedding-ada-002")
    """
    
    def __init__(self, embedding_model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is not installed. Install it with: pip install openai"
            )
        
        self.embedding_model = embedding_model
        self.client = OpenAI()
        
        # Set embedding size based on model
        if embedding_model == "text-embedding-ada-002":
            self.embedding_size = 1536
        elif embedding_model in ["text-embedding-3-small", "text-embedding-3-large"]:
            # These models support dimensions parameter, default to 1536 for small, 3072 for large
            if "small" in embedding_model:
                self.embedding_size = 1536
            else:
                self.embedding_size = 3072
        else:
            # Default to 1536 for unknown models, will be updated on first call
            self.embedding_size = 1536
    
    async def encode_async(self, sentences: Sentences) -> List[List[float]]:
        """Encode a list of sentences into their corresponding sentence embeddings.
        
        Args:
            sentences (Sentences): The list of sentences to be encoded.
            
        Returns:
            List[List[float]]: The list of sentence embeddings, where each embedding is a list of floats.
        """
        import logging
        logger = logging.getLogger("vijil.dome")
        
        loop = asyncio.get_running_loop()
        
        # Run the synchronous OpenAI call in an executor
        def _get_embeddings():
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=sentences
            )
            return [item.embedding for item in response.data]
        
        embeddings = await loop.run_in_executor(None, _get_embeddings)
        
        # Always update embedding_size from actual API response
        if embeddings:
            actual_dim = len(embeddings[0])
            if self.embedding_size != actual_dim:
                logger.warning(
                    f"Embedding dimension mismatch for model '{self.embedding_model}': "
                    f"expected {self.embedding_size}, got {actual_dim}. Updating to actual dimension."
                )
                self.embedding_size = actual_dim
        
        return embeddings
