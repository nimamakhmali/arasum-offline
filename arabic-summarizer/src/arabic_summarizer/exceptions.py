"""
کلاس‌های خطای اختصاصی پروژه. در لایه API به پاسخ‌های HTTP مناسب تبدیل می‌شوند.
"""


class SummarizerBaseException(Exception):
    """کلاس پایه برای همه خطاهای پروژه"""
    pass


class ModelLoadError(SummarizerBaseException):
    """زمانی که فایل مدل یا tokenizer پیدا/لود نشود"""
    pass


class InferenceError(SummarizerBaseException):
    """خطا در حین اجرای مدل (timeout، خطای runtime و ...)"""
    pass


class ValidationError(SummarizerBaseException):
    """طول متن خارج از محدوده مجاز (300-3000 کلمه) یا ورودی نامعتبر"""
    pass


class CacheError(SummarizerBaseException):
    """خطا در خواندن/نوشتن کش"""
    pass
