"""BC-20: Corrupted cache files should not crash loaders."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCorruptedCacheRecovery:
    """Corrupt JSON in cache files should not crash loaders."""

    def test_config_loader_survives_corrupt_cache(self, tmp_path: Path) -> None:
        from vijil_dome.utils.config_loader import load_dome_config_from_s3

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with patch("vijil_dome.utils.config_loader._create_s3_client") as mock_s3:
            mock_client = MagicMock()
            mock_s3.return_value = mock_client

            good_config = {"input-guards": ["prompt-injection"]}
            mock_client.get_object.return_value = {
                "Body": MagicMock(read=lambda: json.dumps(good_config).encode()),
                "ETag": '"abc123"',
            }

            result = load_dome_config_from_s3(
                bucket="test-bucket",
                key="test/config.json",
                cache_dir=str(cache_dir),
                cache_ttl_seconds=9999,
            )
            assert result == good_config

            # Corrupt the cached file
            config_files = list(cache_dir.rglob("config.json"))
            assert len(config_files) == 1
            config_files[0].write_text("{invalid json!!!")

            # Second load should re-download instead of crashing
            result2 = load_dome_config_from_s3(
                bucket="test-bucket",
                key="test/config.json",
                cache_dir=str(cache_dir),
                cache_ttl_seconds=9999,
            )
            assert result2 == good_config

    def test_policy_loader_raises_on_corrupt_local_file(self, tmp_path: Path) -> None:
        from vijil_dome.utils.policy_loader import load_policy_sections_from_file

        bad_file = tmp_path / "policy.json"
        bad_file.write_text("{not valid json!!!")

        with pytest.raises(ValueError, match="Invalid JSON in policy file"):
            load_policy_sections_from_file(str(bad_file))
