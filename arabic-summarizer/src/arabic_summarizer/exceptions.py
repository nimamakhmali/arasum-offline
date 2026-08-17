"""
سلسله‌مراتب خطاهای سفارشی پروژه arasum-offline.

همه خطاها از ArabicSummarizerError ارث می‌برند تا
بتوان همه آن‌ها را با یک except گرفت.
"""


class ArabicSummarizerError(Exception):
    """
    کلاس پایه تمام خطاهای این پروژه.
    خطاهای عمومی غیرقابل دسته‌بندی از این کلاس ارث می‌برند.
    """

    default_message = "خطایی در سیستم خلاصه‌سازی رخ داد."

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.default_message)


# ─────────────────────────────────────────────
# خطاهای مربوط به ورودی
# ─────────────────────────────────────────────


class InvalidInputError(ArabicSummarizerError):
    """
    متن ورودی معتبر نیست.
    معمولاً به دلیل طول خارج از محدوده ۳۰۰ تا ۳۰۰۰ کلمه.

    Attributes:
        word_count: تعداد کلمات متن ورودی
        min_words: حداقل مجاز
        max_words: حداکثر مجاز
    """

    default_message = "متن ورودی خارج از محدوده مجاز است."

    def __init__(
        self,
        message: str = "",
        word_count: int | None = None,
        min_words: int = 300,
        max_words: int = 3000,
    ) -> None:
        if not message and word_count is not None:
            message = (
                f"تعداد کلمات ورودی {word_count} است. "
                f"محدوده مجاز: {min_words} تا {max_words} کلمه."
            )
        super().__init__(message or self.default_message)
        self.word_count = word_count
        self.min_words = min_words
        self.max_words = max_words


class UnsupportedLanguageError(ArabicSummarizerError):
    """
    متن ورودی به زبان عربی نیست یا درصد محتوای عربی کافی نیست.
    """

    default_message = "متن ورودی باید به زبان عربی باشد."

    def __init__(self, message: str = "", detected_language: str = "") -> None:
        if not message and detected_language:
            message = (
                f"زبان شناسایی‌شده '{detected_language}' است. "
                "لطفاً متن عربی وارد کنید."
            )
        super().__init__(message or self.default_message)
        self.detected_language = detected_language


class InvalidRatioError(ArabicSummarizerError):
    """
    نسبت خلاصه‌سازی خارج از محدوده ۰.۱۰ تا ۰.۳۰ است.
    """

    default_message = "نسبت خلاصه‌سازی باید بین ۰.۱۰ و ۰.۳۰ باشد."

    def __init__(self, message: str = "", ratio: float | None = None) -> None:
        if not message and ratio is not None:
            message = (
                f"نسبت داده‌شده {ratio:.2f} است. "
                "مقدار مجاز: بین ۰.۱۰ و ۰.۳۰."
            )
        super().__init__(message or self.default_message)
        self.ratio = ratio


# ─────────────────────────────────────────────
# خطاهای مربوط به مدل
# ─────────────────────────────────────────────


class ModelNotFoundError(ArabicSummarizerError):
    """
    فایل مدل در مسیر مشخص‌شده وجود ندارد.

    Attributes:
        model_path: مسیری که جستجو شد
    """

    default_message = "فایل مدل پیدا نشد."

    def __init__(self, message: str = "", model_path: str = "") -> None:
        if not message and model_path:
            message = f"فایل مدل در مسیر '{model_path}' وجود ندارد."
        super().__init__(message or self.default_message)
        self.model_path = model_path


class ModelLoadError(ArabicSummarizerError):
    """
    بارگذاری مدل با خطا مواجه شد.
    فایل وجود دارد اما خواندن آن ممکن نیست.
    """

    default_message = "بارگذاری مدل با شکست مواجه شد."


# ─────────────────────────────────────────────
# خطاهای مربوط به پردازش
# ─────────────────────────────────────────────


class PreprocessingError(ArabicSummarizerError):
    """
    در حین پیش‌پردازش متن خطایی رخ داد.
    """

    default_message = "پیش‌پردازش متن با خطا مواجه شد."


class InferenceError(ArabicSummarizerError):
    """
    در حین تولید خلاصه توسط مدل خطایی رخ داد.
    """

    default_message = "تولید خلاصه با خطا مواجه شد."


class PostprocessingError(ArabicSummarizerError):
    """
    در حین پس‌پردازش خروجی مدل خطایی رخ داد.
    """

    default_message = "پس‌پردازش خروجی با خطا مواجه شد."


# ─────────────────────────────────────────────
# خطاهای مربوط به کش
# ─────────────────────────────────────────────


class CacheError(ArabicSummarizerError):
    """
    عملیات خواندن یا نوشتن در کش با خطا مواجه شد.
    """

    default_message = "عملیات کش با خطا مواجه شد."