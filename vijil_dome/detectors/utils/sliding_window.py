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

"""Sliding window chunking for HuggingFace tokenizer-based detectors.

When input text exceeds a model's ``max_length``, the text is split into
overlapping token windows so every token appears in at least one chunk.
Each chunk is decoded back to a string that can be fed directly to a
HuggingFace pipeline.
"""

from __future__ import annotations

from typing import List

from transformers import PreTrainedTokenizerBase


def needs_chunking(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    special_token_overhead: int = 2,
) -> bool:
    """Return True if *text* exceeds *max_length* tokens (fast path check)."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids) > (max_length - special_token_overhead)


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
    special_token_overhead: int = 2,
) -> List[str]:
    """Tokenize *text* and, if it exceeds *max_length*, split into overlapping windows.

    Parameters
    ----------
    text:
        The raw input string.
    tokenizer:
        A HuggingFace tokenizer instance.
    max_length:
        Maximum number of tokens per window **including** special tokens.
    stride:
        Step size (in tokens) between successive windows. A stride smaller
        than ``usable`` (= ``max_length - special_token_overhead``) creates
        overlap, which is important for catching content at window boundaries.
    special_token_overhead:
        Number of special tokens the model adds (e.g. [CLS] + [SEP] = 2).

    Returns
    -------
    List[str]
        A list of text chunks.  If the input fits in a single window the
        list contains just the original *text* (fast path — no re-encoding).
    """
    if not text:
        return [text]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    usable = max_length - special_token_overhead

    if len(token_ids) <= usable:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + usable, len(token_ids))
        window_ids = token_ids[start:end]
        chunk = tokenizer.decode(window_ids, skip_special_tokens=True)
        chunks.append(chunk)
        if end >= len(token_ids):
            break
        start += stride

    return chunks
