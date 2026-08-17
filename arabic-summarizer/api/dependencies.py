"""
مدیریت وابستگی‌های مشترک FastAPI - از جمله بارگذاری Singleton مدل.
"""

from functools import lru_cache
from arabic_summarizer import Summarizer
from arabic_summarizer.utils.config_loader import load_config

MODEL_CONFIG = load_config("configs/model_config.yaml")


@lru_cache()
def get_summarizer() -> Summarizer:
    """
    مدل فقط یک‌بار در طول عمر اپلیکیشن بارگذاری می‌شود (نه در هر request)
    تا الزام زمان پاسخ (زیر 15 ثانیه) رعایت شود.
    """
    return Summarizer(
        model_path=MODEL_CONFIG["model"]["production_path"],
        tokenizer_path=MODEL_CONFIG["model"]["tokenizer_path"],
    )
