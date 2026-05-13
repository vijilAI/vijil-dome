"""Tests for embedding index fixes: BC-14 (shared mutable state)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from vijil_dome.embeds import EmbeddingsItem
from vijil_dome.embeds.index.basic import InMemEmbeddingsIndex


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return embedder


# ---------------------------------------------------------------------------
# BC-14: Instances must have independent _items and _embeddings
# ---------------------------------------------------------------------------


def test_instances_have_independent_items() -> None:
    embedder_a = _make_mock_embedder()
    embedder_b = _make_mock_embedder()

    index_a = InMemEmbeddingsIndex(embedder_a)
    index_b = InMemEmbeddingsIndex(embedder_b)

    index_a._items.append(EmbeddingsItem(text="only in a"))

    assert len(index_a._items) == 1
    assert len(index_b._items) == 0
    assert index_a._items is not index_b._items


def test_instances_have_independent_embeddings() -> None:
    embedder_a = _make_mock_embedder()
    embedder_b = _make_mock_embedder()

    index_a = InMemEmbeddingsIndex(embedder_a)
    index_b = InMemEmbeddingsIndex(embedder_b)

    index_a._embeddings.append([0.1, 0.2, 0.3])

    assert len(index_a._embeddings) == 1
    assert len(index_b._embeddings) == 0
    assert index_a._embeddings is not index_b._embeddings
