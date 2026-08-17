import os
import tempfile
from arabic_summarizer.cache.cache_manager import CacheManager


def test_cache_set_and_get():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_cache.db")
        cache = CacheManager(db_path=db_path)

        cache.set("key1", "summary1")
        assert cache.get("key1") == "summary1"


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_cache.db")
        cache = CacheManager(db_path=db_path)

        assert cache.get("nonexistent_key") is None
