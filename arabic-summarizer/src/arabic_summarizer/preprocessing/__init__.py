"""
خط‌لوله پیش‌پردازش متون عربی.

این ماژول سه لایه پردازش را به ترتیب اجرا می‌کند:
1. ArabicTextCleaner   : حذف نویز و کاراکترهای غیرضروری
2. ArabicNormalizer    : نرمال‌سازی حروف و اعراب
3. NLTK                : حذف stopwords و tokenization (اگر نصب باشد)
4. CamelTools          : نرمال‌سازی تکمیلی (اگر نصب باشد)

استفاده:
    from arabic_summarizer.preprocessing import ArabicPreprocessingPipeline

    pipeline = ArabicPreprocessingPipeline()
    result = pipeline.run("متن عربی...")
    print(result.cleaned_text)
    print(result.word_count)
    print(result.nltk_used)
    print(result.camel_tools_used)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arabic_summarizer.exceptions import InvalidInputError, PreprocessingError
from arabic_summarizer.preprocessing.cleaner import ArabicTextCleaner
from arabic_summarizer.preprocessing.normalizer import ArabicNormalizer
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """
    نتیجه خروجی pipeline پیش‌پردازش.

    Attributes:
        cleaned_text     : متن پردازش‌شده نهایی آماده برای مدل
        word_count       : تعداد کلمات بعد از پاک‌سازی
        camel_tools_used : آیا CamelTools در این پردازش استفاده شد
        nltk_used        : آیا NLTK در این پردازش استفاده شد
        removed_stopwords: تعداد stopword های حذف‌شده توسط NLTK
    """

    cleaned_text: str
    word_count: int
    camel_tools_used: bool
    nltk_used: bool
    removed_stopwords: int = field(default=0)


class ArabicPreprocessingPipeline:
    """
    خط‌لوله یکپارچه پیش‌پردازش متون عربی.

    این کلاس تمام مراحل پیش‌پردازش را به ترتیب صحیح اجرا می‌کند.
    اگر NLTK یا CamelTools نصب نباشند، سیستم بدون خطا ادامه می‌دهد
    و از ماژول‌های داخلی استفاده می‌کند.
    """

    def __init__(
        self,
        min_words: int = 300,
        max_words: int = 3000,
        normalize_teh: bool = False,
        remove_stopwords: bool = True,
        use_camel_tools: bool = True,
        use_nltk: bool = True,
    ) -> None:
        """
        Args:
            min_words        : حداقل کلمات مجاز در ورودی
            max_words        : حداکثر کلمات مجاز در ورودی
            normalize_teh    : آیا تاء مربوطه نرمال شود
            remove_stopwords : آیا stopword های عربی حذف شوند
                               توجه: برای خلاصه‌سازی abstractive
                               توصیه نمی‌شود چون ممکن است معنا تغییر کند.
                               پیش‌فرض False برای این دلیل است.
            use_camel_tools  : آیا از CamelTools استفاده شود
            use_nltk         : آیا از NLTK استفاده شود
        """
        self._cleaner = ArabicTextCleaner(min_words=min_words, max_words=max_words)
        self._normalizer = ArabicNormalizer()
        self._normalize_teh = normalize_teh
        self._remove_stopwords = remove_stopwords

        # بارگذاری lazy - فقط اگر نیاز باشد
        self._nltk_stopwords: set[str] | None = None
        self._camel_normalizer = None

        if use_nltk:
            self._nltk_stopwords = self._try_load_nltk()

        if use_camel_tools:
            self._camel_normalizer = self._try_load_camel_tools()

    # ── بارگذاری ابزارهای اختیاری ───────────────────────────────

    @staticmethod
    def _try_load_nltk() -> set[str] | None:
        """
        بارگذاری stopwords عربی از NLTK.

        NLTK برای این پروژه دو کاربرد دارد:
        1. دریافت لیست stopwords عربی
        2. tokenization کمکی برای شمارش دقیق‌تر کلمات

        اگر NLTK نصب نباشد یا داده‌های آن دانلود نشده باشند،
        None برمی‌گرداند و هشدار لاگ می‌کند.
        """
        try:
            import nltk
            from nltk.corpus import stopwords

            # تلاش برای دانلود اگر وجود نداشت
            try:
                arabic_stopwords = set(stopwords.words("arabic"))
            except LookupError:
                logger.info("دانلود stopwords عربی از NLTK ...")
                nltk.download("stopwords", quiet=True)
                nltk.download("punkt", quiet=True)
                arabic_stopwords = set(stopwords.words("arabic"))

            logger.info(
                "NLTK با موفقیت بارگذاری شد. تعداد stopwords: %d",
                len(arabic_stopwords),
            )
            return arabic_stopwords

        except ImportError:
            logger.warning(
                "NLTK نصب نیست. stopword removal غیرفعال می‌شود. "
                "برای نصب: pip install nltk"
            )
            return None
        except Exception as exc:
            logger.warning("خطا در بارگذاری NLTK: %s", exc)
            return None

    @staticmethod
    def _try_load_camel_tools():
        """
        بارگذاری ابزارهای نرمال‌سازی CamelTools.

        CamelTools ابزار تخصصی پردازش متن عربی است که
        normalize های دقیق‌تری نسبت به regex ساده دارد.

        اگر نصب نباشد None برمی‌گرداند.
        """
        try:
            from camel_tools.utils.normalize import (
                normalize_unicode,
                normalize_alef_ar,
                normalize_alef_maksura_ar,
            )

            def camel_normalize(text: str) -> str:
                """ترکیب normalize های CamelTools."""
                text = normalize_unicode(text)
                text = normalize_alef_ar(text)
                text = normalize_alef_maksura_ar(text)
                return text

            logger.info("CamelTools با موفقیت بارگذاری شد.")
            return camel_normalize

        except ImportError:
            logger.warning(
                "CamelTools نصب نیست. از normalizer داخلی استفاده می‌شود. "
                "برای نصب: pip install camel-tools"
            )
            return None
        except Exception as exc:
            logger.warning("خطا در بارگذاری CamelTools: %s", exc)
            return None

    # ── پردازش stopwords ─────────────────────────────────────────

    def _apply_stopword_removal(self, text: str) -> tuple[str, int]:
        """
        حذف stopwords عربی از متن.

        نکته مهم: این عملیات برای خلاصه‌سازی extractive مفیدتر است.
        برای خلاصه‌سازی abstractive (که این پروژه از آن استفاده می‌کند)
        معمولاً stopword removal روی متن ورودی انجام نمی‌شود چون
        مدل ترانسفورمر خودش اهمیت کلمات را یاد می‌گیرد.

        با این حال برای آماده‌سازی دیتاست آموزشی مفید است.

        Returns:
            (متن بدون stopword، تعداد کلمات حذف‌شده)
        """
        if self._nltk_stopwords is None or not self._remove_stopwords:
            return text, 0

        words = text.split()
        filtered = [w for w in words if w not in self._nltk_stopwords]
        removed_count = len(words) - len(filtered)

        return " ".join(filtered), removed_count

    # ── متد اصلی ────────────────────────────────────────────────

    def run(self, text: str, apply_stopword_removal: bool = False) -> PreprocessingResult:
        """
        اجرای کامل pipeline روی متن ورودی.

        Args:
            text                  : متن خام عربی
            apply_stopword_removal: آیا stopword removal اجرا شود.
                                    پیش‌فرض False چون برای مدل abstractive
                                    توصیه نمی‌شود. فقط برای آماده‌سازی
                                    دیتاست آموزشی True کنید.

        Returns:
            PreprocessingResult با تمام اطلاعات پردازش

        Raises:
            InvalidInputError : اگر طول متن خارج از محدوده باشد
            PreprocessingError: اگر خطای غیرمنتظره‌ای رخ دهد
        """
        try:
            # ── مرحله ۱: پاک‌سازی و اعتبارسنجی ──────────────────
            cleaned, word_count = self._cleaner.clean_and_validate(text)
            logger.debug("مرحله ۱ کامل شد - پاک‌سازی. کلمات: %d", word_count)

            # ── مرحله ۲: نرمال‌سازی داخلی ─────────────────────────
            normalized = self._normalizer.normalize(
                cleaned, normalize_teh=self._normalize_teh
            )
            logger.debug("مرحله ۲ کامل شد - نرمال‌سازی داخلی.")

            # ── مرحله ۳: CamelTools (اگر موجود باشد) ──────────────
            camel_used = False
            if self._camel_normalizer is not None:
                normalized = self._camel_normalizer(normalized)
                camel_used = True
                logger.debug("مرحله ۳ کامل شد - CamelTools.")

            # ── مرحله ۴: NLTK Stopword Removal (اختیاری) ──────────
            removed_count = 0
            nltk_used = self._nltk_stopwords is not None

            if apply_stopword_removal and self._remove_stopwords:
                normalized, removed_count = self._apply_stopword_removal(normalized)
                logger.debug(
                    "مرحله ۴ کامل شد - NLTK. stopwords حذف‌شده: %d",
                    removed_count,
                )

                # شمارش مجدد کلمات بعد از حذف stopword
                word_count = self._cleaner.count_words(normalized)

            return PreprocessingResult(
                cleaned_text=normalized,
                word_count=word_count,
                camel_tools_used=camel_used,
                nltk_used=nltk_used,
                removed_stopwords=removed_count,
            )

        except InvalidInputError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                f"خطای غیرمنتظره در pipeline پیش‌پردازش: {exc}"
            ) from exc

    # ── اطلاعات pipeline ─────────────────────────────────────────

    def get_info(self) -> dict:
        """
        اطلاعات وضعیت pipeline را برمی‌گرداند.
        برای debugging و health check مفید است.
        """
        return {
            "nltk_available": self._nltk_stopwords is not None,
            "nltk_stopwords_count": (
                len(self._nltk_stopwords) if self._nltk_stopwords else 0
            ),
            "camel_tools_available": self._camel_normalizer is not None,
            "stopword_removal_enabled": self._remove_stopwords,
            "teh_normalization_enabled": self._normalize_teh,
        }