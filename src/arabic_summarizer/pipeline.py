"""
Pipeline اصلی پردازش: یک تابع ساده Composition، نه کلاس اضافه.
ورودی این تابع باید از قبل clean و normalize شده باشد
(این کار در summarizer.py برای جلوگیری از تکرار انجام می‌شود).
"""

from .postprocessing.formatter import format_summary


def run(normalized_text: str, ratio: float, engine) -> str:
    raw_summary = engine.generate(normalized_text, ratio=ratio)
    final_summary = format_summary(raw_summary)
    return final_summary
