"""
پس‌پردازش خروجی مدل: اصلاح فاصله‌گذاری، جمله‌بندی و فرمت نهایی.
"""

import re


def format_summary(raw_summary: str) -> str:
    """پاکسازی نهایی متن خلاصه‌شده قبل از بازگشت به کاربر"""
    summary = re.sub(r"\s+", " ", raw_summary).strip()
    if summary and summary[-1] not in ".!؟":
        summary += "."
    return summary
