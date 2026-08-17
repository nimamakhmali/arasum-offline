"""
خط‌لوله پیش‌پردازش متون عربی.

استفاده ساده:
    from arabic_summarizer.preprocessing import ArabicPreprocessingPipeline

    pipeline = ArabicPreprocessingPipeline()
    result = pipeline.run("متن عربی...")
    print(result.cleaned_text)
"""

from __future__ import annotations

from dataclasses import dataclass

from arabic_summarizer.exceptions import PreprocessingError
from arabic_summarizer.preprocessing.cleaner import ArabicTextCleaner
from arabic_summarizer.preprocessing.normalizer import ArabicNormalizer
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """نتیجه خروجی pipeline پیش‌پردازش."""

    cleaned_text: str
    word_count: int
    camel_tools_used: bool


class ArabicPreprocessingPipeline:
    """
    خط‌لوله یکپارچه پیش‌پردازش متون عربی.

    مراحل به ترتیب:
    1. ArabicTextCleaner: حذف نویز و کاراکترهای غیرضروری
    2. ArabicNormalizer: نرمال‌سازی حروف و اعراب
    3. CamelTools (اختیاری): نرمال‌سازی تکمیلی اگر نصب باشد

    اگر CamelTools نصب نباشد، سیستم بدون خطا ادامه می‌دهد.
    """

    def __init__(
        self,
        min_words: int = 300,
        max_words: int = 3000,
        normalize_teh: bool = False,
        use_camel_tools: bool = True,
    ) -> None:
        """
        Args:
            min_words: حداقل کلمات مجاز
            max_words: حداکثر کلمات مجاز
            normalize_teh: آیا تاء مربوطه نرمال شود
            use_camel_tools: آیا از CamelTools استفاده شود
        """
        self._cleaner = ArabicTextCleaner(min_words=min_words, max_words=max_words)
        self._normalizer = ArabicNormalizer()
        self._normalize_teh = normalize_teh
        self._camel_normalizer = self._try_load_camel_tools(use_camel_tools)

    @staticmethod
    def _try_load_camel_tools(enabled: bool):
        """
        تلاش برای بارگذاری CamelTools.
        اگر نصب نباشد None برمی‌گرداند و هشدار لاگ می‌کند.
        """
        if not enabled:
            return None
        try:
            from camel_tools.utils.normalize import (
                normalize_unicode,
                normalize_alef_maksura_ar,
                normalize_alef_ar,
                normalize_teh_marbuta_ar,
            )
            logger.info("CamelTools با موفقیت بارگذاری شد.")

            # یک callable ساده که همه normalize های camel را اجرا می‌کند
            def camel_normalize(text: str) -> str:
                text = normalize_unicode(text)
                text = normalize_alef_ar(text)
                text = normalize_alef_maksura_ar(text)
                return text

            return camel_normalize

        except ImportError:
            logger.warning(
                "CamelTools نصب نیست. "
                "از normalizer داخلی استفاده می‌شود. "
                "برای نصب: pip install camel-tools"
            )
            return None

    def run(self, text: str) -> PreprocessingResult:
        """
        اجرای کامل pipeline روی متن ورودی.

        Args:
            text: متن خام عربی

        Returns:
            PreprocessingResult شامل متن پردازش‌شده و آمار

        Raises:
            PreprocessingError: اگر pipeline با خطای غیرمنتظره مواجه شود
            InvalidInputError: اگر طول متن خارج از محدوده باشد
        """
        try:
            # مرحله ۱: پاک‌سازی و اعتبارسنجی
            cleaned, word_count = self._cleaner.clean_and_validate(text)
            logger.debug("پاک‌سازی کامل شد. تعداد کلمات: %d", word_count)

            # مرحله ۲: نرمال‌سازی داخلی
            normalized = self._normalizer.normalize(
                cleaned, normalize_teh=self._normalize_teh
            )
            logger.debug("نرمال‌سازی داخلی کامل شد.")

            # مرحله ۳: CamelTools (اگر موجود باشد)
            camel_used = False
            if self._camel_normalizer is not None:
                normalized = self._camel_normalizer(normalized)
                camel_used = True
                logger.debug("نرمال‌سازی CamelTools کامل شد.")

            return PreprocessingResult(
                cleaned_text=normalized,
                word_count=word_count,
                camel_tools_used=camel_used,
            )

        except Exception as exc:
            # InvalidInputError را دوباره raise می‌کنیم بدون wrap
            from arabic_summarizer.exceptions import InvalidInputError
            if isinstance(exc, InvalidInputError):
                raise
            raise PreprocessingError(
                f"خطای غیرمنتظره در pipeline: {exc}"
            ) from exc