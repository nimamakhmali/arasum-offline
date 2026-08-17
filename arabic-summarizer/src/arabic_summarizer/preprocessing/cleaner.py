"""
حذف اعراب، نویز و نشانه‌های غیرضروری از متن عربی.
TODO: پیاده‌سازی با کمک CamelTools یا regex اختصاصی عربی.
"""

import re


def remove_diacritics(text: str) -> str:
    """حذف اعراب (تشکیل) از متن عربی"""
    arabic_diacritics = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
    return re.sub(arabic_diacritics, "", text)


def remove_noise(text: str) -> str:
    """حذف کاراکترهای غیرضروری، لینک‌ها، فاصله‌های تکراری و ..."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """تابع اصلی pipeline پاکسازی متن"""
    text = remove_diacritics(text)
    text = remove_noise(text)
    return text
