"""
تست‌های واحد برای ConfigLoader.

اجرا:
    pytest tests/unit/test_config_loader.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from arabic_summarizer.utils.config_loader import ConfigLoader


@pytest.fixture
def temp_configs_dir(tmp_path: Path) -> Path:
    """یک پوشه موقت با فایل‌های config می‌سازد."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # فایل تست
    test_config = {
        "model": {
            "name": "AraBART",
            "path": "/models/arabart",
        },
        "generation": {
            "num_beams": 4,
            "max_length": 256,
        },
    }
    with open(configs_dir / "test_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    return configs_dir


@pytest.fixture
def loader(temp_configs_dir: Path, monkeypatch) -> ConfigLoader:
    """
    یک ConfigLoader با پوشه موقت می‌سازد.
    Singleton را reset می‌کند تا تست‌ها مستقل باشند.
    """
    # Reset singleton برای هر تست
    ConfigLoader._instance = None
    ConfigLoader._cache = {}

    monkeypatch.setenv("CONFIGS_DIR", str(temp_configs_dir))
    instance = ConfigLoader()
    return instance


class TestConfigLoader:

    def test_singleton_pattern(self, loader):
        """دو instance باید یکی باشند."""
        second = ConfigLoader()
        assert loader is second

    def test_get_simple_value(self, loader):
        """مقدار ساده باید درست برگردد."""
        assert loader.get("test_config", "model.name") == "AraBART"

    def test_get_nested_value(self, loader):
        """مقدار تو در تو باید درست برگردد."""
        assert loader.get("test_config", "generation.num_beams") == 4

    def test_get_nonexistent_key_returns_default(self, loader):
        """کلید ناموجود باید مقدار default برگرداند."""
        result = loader.get("test_config", "model.nonexistent", default="fallback")
        assert result == "fallback"

    def test_get_nonexistent_key_returns_none_by_default(self, loader):
        """بدون default، کلید ناموجود باید None برگرداند."""
        result = loader.get("test_config", "not.exists")
        assert result is None

    def test_get_nonexistent_file_returns_default(self, loader):
        """فایل ناموجود نباید crash کند و default برگرداند."""
        result = loader.get("nonexistent_file", "any.key", default="safe")
        assert result == "safe"

    def test_get_section(self, loader):
        """get_section باید دیکشنری کامل بخش را برگرداند."""
        section = loader.get_section("test_config", "model")
        assert isinstance(section, dict)
        assert section["name"] == "AraBART"
        assert section["path"] == "/models/arabart"

    def test_get_section_nonexistent_returns_empty(self, loader):
        """بخش ناموجود باید دیکشنری خالی برگرداند."""
        section = loader.get_section("test_config", "nonexistent")
        assert section == {}

    def test_reload_clears_cache(self, loader):
        """reload باید کش را پاک کند."""
        # اول load کن
        loader.get("test_config", "model.name")
        assert "test_config" in loader._cache

        loader.reload()
        assert "test_config" not in loader._cache