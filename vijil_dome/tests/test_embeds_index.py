"""Tests for embedding index fixes: BC-14 (shared mutable state), BC-19 (FAISS dimension validation)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


# ---------------------------------------------------------------------------
# BC-19: FAISS index dimension validation
# ---------------------------------------------------------------------------


def test_faiss_dimension_mismatch_raises() -> None:
    faiss = pytest.importorskip("faiss")
    from vijil_dome.embeds.index.faiss_index import FaissEmbeddingsIndex

    mock_embedder = MagicMock()
    mock_embedder.embedding_size = 384

    index = FaissEmbeddingsIndex(embedder=mock_embedder)

    fake_faiss_index = faiss.IndexFlatL2(1536)
    fake_faiss_index.add(
        __import__("numpy").random.randn(5, 1536).astype("float32")
    )

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        faiss.write_index(fake_faiss_index, f.name)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="does not match"):
            index.load_from_file(tmp_path)
    finally:
        os.unlink(tmp_path)
