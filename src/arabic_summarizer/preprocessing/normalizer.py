"""
نرمال‌سازی حروف و متن عربی.

این ماژول هیچ وابستگی خارجی ندارد و فقط از re استاندارد استفاده می‌کند.
ترتیب اجرای عملیات مهم است و نباید تغییر کند.
"""

from __future__ import annotations

import re


class ArabicNormalizer:
    """
    نرمال‌سازی متن عربی برای یکسان‌سازی اشکال مختلف حروف.

    عملیات به ترتیب:
    1. نرمال‌سازی انواع الف
    2. نرمال‌سازی تاء مربوطه (اختیاری)
    3. نرمال‌سازی یاء و اشکال مختلف آن
    4. حذف اعراب (حرکات)
    5. حذف کشیده (تطویل)
    6. یکسان‌سازی فاصله‌ها
    """

    # ── الف ────────────────────────────────────────────────────
    # أ (U+0623)، إ (U+0625)، آ (U+0622)، ٱ (U+0671) → ا (U+0627)
    _ALEF_PATTERN = re.compile(r"[أإآٱ]")

    # ── تاء مربوطه ─────────────────────────────────────────────
    # ة (U+0629) → ه (U+0647) فقط انتهای کلمه
    _TEH_MARBUTA_PATTERN = re.compile(r"ة\b")

    # ── یاء - اصلاح شده ────────────────────────────────────────
    # سه شکل مختلف یاء را یکسان می‌کنیم:
    # ى  (U+0649 - الف مقصوره / yeh without dots)
    # ی  (U+06CC - Farsi Yeh)
    # ي  (U+064A - Arabic Yeh) → شکل استاندارد
    #
    # توضیح: در متون عربی دیجیتال گاهی این سه با هم مخلوط می‌شوند.
    # یکسان‌سازی آن‌ها کیفیت tokenization را بهبود می‌دهد.
    _YEH_PATTERN = re.compile(r"[ىی]")

    # ── اعراب و حرکات ──────────────────────────────────────────
    # U+064B فتحتان تا U+065F شامل تمام حرکات استاندارد
    # U+0670 الف خنجریه (superscript alef)
    _DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")

    # ── کشیده / تطویل ──────────────────────────────────────────
    # ـ (U+0640)
    _TATWEEL_PATTERN = re.compile(r"\u0640")

    # ── فاصله‌های اضافه ─────────────────────────────────────────
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    def normalize_alef(self, text: str) -> str:
        """
        تبدیل تمام اشکال الف به شکل ساده.
        أحمد / إبراهيم / آدم → احمد / ابراهيم / ادم
        """
        return self._ALEF_PATTERN.sub("ا", text)

    def normalize_teh_marbuta(self, text: str) -> str:
        """
        تبدیل تاء مربوطه به هاء در انتهای کلمه.
        مدرسة → مدرسه

        نکته: این تبدیل اختیاری است و بسته به کاربرد می‌تواند غیرفعال شود.
        برای متون قرآنی توصیه نمی‌شود.
        """
        return self._TEH_MARBUTA_PATTERN.sub("ه", text)

    def normalize_yeh(self, text: str) -> str:
        """
        یکسان‌سازی اشکال مختلف یاء به شکل استاندارد عربی.

        اشکال نرمال‌شده:
        - ى  (U+0649 - الف مقصوره) → ي
        - ی  (U+06CC - Farsi Yeh)  → ي

        دلیل: در متون عربی دیجیتال، این سه شکل گاهی با هم مخلوط
        می‌شوند و باعث عدم تطابق در tokenization و جستجو می‌شوند.

        مثال:
            موسى → موسي
            يُحيى → يُحيي
            می (فارسی) → مي
        """
        return self._YEH_PATTERN.sub("ي", text)

    def remove_diacritics(self, text: str) -> str:
        """
        حذف تمام اعراب و حرکات.
        مَدْرَسَة → مدرسة
        كِتَابٌ → كتاب
        """
        return self._DIACRITICS_PATTERN.sub("", text)

    def remove_tatweel(self, text: str) -> str:
        """
        حذف نویسه کشیده که برای زیبایی به کلمات اضافه می‌شود.
        ك\u0640\u0640تاب → كتاب

        توجه: فقط کاراکتر U+0640 حذف می‌شود.
        الف‌های تکراری (كتاااب) تغییر نمی‌کنند.
        """
        return self._TATWEEL_PATTERN.sub("", text)

    def normalize_whitespace(self, text: str) -> str:
        """
        تبدیل چند فاصله یا whitespace متوالی به یک فاصله.
        حذف فاصله ابتدا و انتهای متن.
        """
        return self._WHITESPACE_PATTERN.sub(" ", text).strip()

    def normalize(self, text: str, normalize_teh: bool = False) -> str:
        """
        اجرای تمام مراحل نرمال‌سازی به ترتیب صحیح.

        ترتیب اجرا:
        1. حذف اعراب (قبل از هر چیز تا تداخل نداشته باشد)
        2. حذف تطویل
        3. نرمال‌سازی الف
        4. نرمال‌سازی یاء (شامل Farsi Yeh)
        5. نرمال‌سازی تاء مربوطه (اختیاری)
        6. یکسان‌سازی فاصله‌ها

        Args:
            text: متن عربی خام
            normalize_teh: آیا تاء مربوطه هم نرمال شود؟
                          برای متون خبری False توصیه می‌شود.

        Returns:
            متن نرمال‌شده
        """
        text = self.remove_diacritics(text)
        text = self.remove_tatweel(text)
        text = self.normalize_alef(text)
        text = self.normalize_yeh(text)
        if normalize_teh:
            text = self.normalize_teh_marbuta(text)
        text = self.normalize_whitespace(text)
        return text