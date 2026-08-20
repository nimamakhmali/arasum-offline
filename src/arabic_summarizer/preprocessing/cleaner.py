"""
پاک‌سازی متون عربی از نویز، تگ‌ها و محتوای غیرضروری.

این ماژول هیچ وابستگی خارجی ندارد.
ترتیب اجرای عملیات مهم است.
"""

from __future__ import annotations

import re

from arabic_summarizer.exceptions import InvalidInputError
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

# محدوده پیش‌فرض طول ورودی مجاز
_DEFAULT_MIN_WORDS = 300
_DEFAULT_MAX_WORDS = 3000


class ArabicTextCleaner:
    """
    پاک‌سازی متن عربی از نویز برای آماده‌سازی ورودی مدل.

    عملیات به ترتیب:
    1. حذف URL
    2. حذف تگ‌های HTML
    3. حذف کاراکترهای غیرعربی (با حفظ اعداد و نشانه‌گذاری)
    4. نرمال‌سازی نشانه‌گذاری تکراری
    5. حذف خطوط خالی اضافه
    6. حذف فاصله‌های اضافه
    """

    # ── URL ────────────────────────────────────────────────────
    _URL_PATTERN = re.compile(
        r"https?://\S+|www\.\S+|ftp://\S+",
        re.IGNORECASE,
    )

    # ── HTML ────────────────────────────────────────────────────
    _HTML_PATTERN = re.compile(r"<[^>]+>")

    # ── کاراکترهای مجاز ─────────────────────────────────────────
    # حروف عربی: U+0600 تا U+06FF
    # حروف عربی تکمیلی: U+0750 تا U+077F
    # اعداد عربی-هندی: ۰-۹ (U+0660 تا U+0669)
    # اعداد لاتین: 0-9
    # نشانه‌گذاری رایج: نقطه، ویرگول، علامت سوال، علامت تعجب
    # نشانه‌گذاری عربی: ، ؛ ؟ «»
    # فاصله و newline
    _ALLOWED_CHARS_PATTERN = re.compile(
        r"[^\u0600-\u06FF\u0750-\u077F"
        r"0-9\u0660-\u0669"
        r"\s"
        r".,!?،؛؟«»\-\(\)\[\]\"\':"
        r"]"
    )

    # ── نشانه‌گذاری تکراری ──────────────────────────────────────
    _REPEATED_PUNCT_PATTERN = re.compile(r"([!?،.؟])\1{2,}")
    _ELLIPSIS_PATTERN = re.compile(r"\.{3,}")

    # ── خطوط خالی اضافه ─────────────────────────────────────────
    _MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

    # ── فاصله‌های اضافه ─────────────────────────────────────────
    _MULTI_SPACE_PATTERN = re.compile(r" {2,}")

    def __init__(
        self,
        min_words: int = _DEFAULT_MIN_WORDS,
        max_words: int = _DEFAULT_MAX_WORDS,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words

    # ── متدهای پاک‌سازی ─────────────────────────────────────────

    def remove_urls(self, text: str) -> str:
        """حذف تمام آدرس‌های اینترنتی."""
        return self._URL_PATTERN.sub(" ", text)

    def remove_html_tags(self, text: str) -> str:
        """حذف تگ‌های HTML. برای متونی که از وب scrappe شده‌اند."""
        return self._HTML_PATTERN.sub(" ", text)

    def remove_non_arabic(self, text: str) -> str:
        """
        حذف کاراکترهای خارج از محدوده مجاز.
        اعداد، نشانه‌گذاری رایج و فاصله حفظ می‌شوند.
        """
        return self._ALLOWED_CHARS_PATTERN.sub(" ", text)

    def normalize_punctuation(self, text: str) -> str:
        """
        یکسان‌سازی نشانه‌گذاری تکراری.
        !!! → !
        ... → .
        """
        text = self._REPEATED_PUNCT_PATTERN.sub(r"\1", text)
        text = self._ELLIPSIS_PATTERN.sub(".", text)
        return text

    def remove_extra_newlines(self, text: str) -> str:
        """تبدیل سه یا بیشتر خط خالی متوالی به دو خط."""
        return self._MULTI_NEWLINE_PATTERN.sub("\n\n", text)

    def remove_extra_spaces(self, text: str) -> str:
        """حذف فاصله‌های اضافه."""
        return self._MULTI_SPACE_PATTERN.sub(" ", text).strip()

    # ── متد اصلی ────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        اجرای تمام مراحل پاک‌سازی به ترتیب صحیح.

        Args:
            text: متن خام

        Returns:
            متن پاک‌شده آماده برای نرمال‌سازی
        """
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_non_arabic(text)
        text = self.normalize_punctuation(text)
        text = self.remove_extra_newlines(text)
        text = self.remove_extra_spaces(text)
        return text

    # ── validation ──────────────────────────────────────────────

    def count_words(self, text: str) -> int:
        """تعداد کلمات متن را برمی‌گرداند."""
        return len(text.split())

    def validate_length(self, text: str) -> int:
        """
        طول متن را بررسی می‌کند.

        Args:
            text: متن برای بررسی

        Returns:
            تعداد کلمات

        Raises:
            InvalidInputError: اگر تعداد کلمات خارج از محدوده باشد
        """
        word_count = self.count_words(text)

        if word_count < self.min_words or word_count > self.max_words:
            logger.warning(
                "متن ورودی %d کلمه دارد (محدوده: %d تا %d)",
                word_count,
                self.min_words,
                self.max_words,
            )
            raise InvalidInputError(word_count=word_count)

        return word_count

    def clean_and_validate(self, text: str) -> tuple[str, int]:
        """
        پاک‌سازی و اعتبارسنجی طول متن.

        Args:
            text: متن خام

        Returns:
            tuple شامل (متن پاک‌شده، تعداد کلمات)

        Raises:
            InvalidInputError: اگر بعد از پاک‌سازی طول نامعتبر باشد
        """
        cleaned = self.clean(text)
        word_count = self.validate_length(cleaned)
        return cleaned, word_count