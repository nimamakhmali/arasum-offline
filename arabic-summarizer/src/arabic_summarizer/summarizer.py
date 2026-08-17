"""
Facade اصلی کتابخانه. تمام منطق cache + pipeline از اینجا هماهنگ می‌شود.
"""

from .pipeline import run as run_pipeline
from .cache.cache_manager import CacheManager
from .utils.hashing import generate_cache_key
from .preprocessing.cleaner import clean_text
from .preprocessing.normalizer import normalize_text
from .inference.onnx_engine import ONNXEngine
from .exceptions import CacheError


class Summarizer:
    def __init__(self, model_path: str, tokenizer_path: str = None, cache_enabled: bool = True):
        self.engine = ONNXEngine(model_path, tokenizer_path)
        self.cache_enabled = cache_enabled
        self.cache = CacheManager() if cache_enabled else None

    def summarize(self, text: str, ratio: float = 0.2) -> str:
        normalized = normalize_text(clean_text(text))

        key = None
        if self.cache_enabled:
            try:
                key = generate_cache_key(normalized, ratio)
                cached = self.cache.get(key)
                if cached:
                    return cached
            except Exception as e:
                raise CacheError(str(e))

        result = run_pipeline(normalized, ratio, self.engine)

        if self.cache_enabled and key:
            try:
                self.cache.set(key, result)
            except Exception as e:
                raise CacheError(str(e))

        return result
