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
import logging
from vijil_dome.embeds import EmbeddingsItem
from vijil_dome.embeds.embedder import Embedder
from vijil_dome.embeds.index.basic import InMemEmbeddingsIndex, AnnoyEmbeddingsIndex


async def main():
    # Define some example texts to embed
    items = [
        ["The quick brown fox jumps over the lazy dog", "26", "1"],
        ["A journey of a thousand miles begins with a single step", "q1", "42"],
        ["To be or not to be, that is the question", "p1", "3"],
        ["All that glitters is not gold", "s1", "4"],
        ["The only thing we have to fear is fear itself", "l1", "5"],
    ]

    embed_items = [
        EmbeddingsItem(text=item[0], meta={"tags": item[1], "id": item[2]})
        for item in items
    ]

    # EMBEDDINGS EXAMPLE

    # Select an engine and model
    embeddings_engine = "FastEmbed"
    # embeddings_engine = "SentenceTransformers"
    embeddings_model = "all-MiniLM-L6-v2"

    embedder = Embedder(
        embedding_engine=embeddings_engine, embedding_model=embeddings_model
    )
    print("Embedder initialized")
    embeds = await embedder.embeddings("This is an emebdding")
    print(f"Embeddings size: {len(embeds[0])}")
    print(f"Embeddings: {embeds[0][:4]}")

    # NEAREST NEIGHBORS EXAMPLE

    # Initialize the BasicEmbeddingsIndex
    index = InMemEmbeddingsIndex(embedder=embedder)
    index = AnnoyEmbeddingsIndex(embedder=embedder)

    # Add items to the index
    await index.add_items(embed_items)
    # Build the index
    await index.build()

    # Perform a search
    import time

    st_time = time.time()
    query = "The quick brown fox"
    embeds_result = await index.nearest_neighbor(query, k=3)

    # Print the search results
    print(f"Search results for query '{query}':")
    for result in embeds_result:
        print(result[0])
    print(f"Time to search: {time.time() - st_time}\n\n")

    # Perform a search with distance
    import time

    st_time = time.time()
    query = "fear the feet of gold"
    embeds_result = await index.nearest_neighbor(query, k=3, with_distance=True)

    # Print the search results
    print(f"Search results for query '{query}':")
    for result in embeds_result:
        print(f"Item: {result[0]}, Distance: {result[1]}")
    print(f"Time to search: {time.time() - st_time}")


# Run the main function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
