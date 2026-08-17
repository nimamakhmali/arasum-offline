"""
تولید کلید کش از متنِ نرمال‌شده (نه متن خام) برای افزایش Cache Hit Rate.
"""

import hashlib


def generate_cache_key(normalized_text: str, ratio: float) -> str:
    raw = f"{normalized_text.strip()}|{ratio}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
