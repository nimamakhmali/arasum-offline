"""
مقایسه چند مدل پایه (AraBART، AraT5 و ...) روی دیتاست‌های EASC/AraBench
قبل از شروع فرآیند Fine-tuning جدی.

TODO:
1. بارگذاری چند مدل کاندید
2. اجرای inference روی نمونه‌ای از دیتاست
3. محاسبه ROUGE برای هرکدام
4. ثبت نتایج در docs/model_selection_report.md
"""

CANDIDATE_MODELS = [
    "moussaKam/AraBART",
    "UBC-NLP/AraT5-base",
    # سایر گزینه‌ها ...
]

if __name__ == "__main__":
    print("TODO: پیاده‌سازی فرآیند مقایسه مدل‌ها")
