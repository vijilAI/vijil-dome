"""BC-13: Import errors should surface at construction, not at scan time."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLazyImportSurfacing:
    """Optional dependency errors surface at construction, not deep in a scan."""

    def test_perspective_soft_gate(self) -> None:
        import vijil_dome.detectors.methods.perspective as mod
        assert hasattr(mod, "_PERSPECTIVE_AVAILABLE")

    def test_faiss_stub_raises_import_error(self) -> None:
        from vijil_dome.embeds.index import _FAISS_AVAILABLE

        if not _FAISS_AVAILABLE:
            from vijil_dome.embeds.index import FaissEmbeddingsIndex

            with pytest.raises(ImportError, match="faiss-cpu"):
                FaissEmbeddingsIndex(embedder=MagicMock())
