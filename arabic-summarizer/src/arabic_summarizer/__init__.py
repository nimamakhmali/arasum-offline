"""
arabic_summarizer - ماژول آفلاین خلاصه‌سازی متون عربی

استفاده سریع:
    from arabic_summarizer.preprocessing import ArabicPreprocessingPipeline

    pipeline = ArabicPreprocessingPipeline()
    result = pipeline.run("متن عربی...")
    print(result.cleaned_text)
"""

__version__ = "0.1.0"
__author__ = "BitForge Team"
__description__ = "Offline Arabic Text Summarization Module"

from arabic_summarizer.exceptions import (
    ArabicSummarizerError,
    InvalidInputError,
    InvalidRatioError,
    UnsupportedLanguageError,
    ModelNotFoundError,
    ModelLoadError,
    PreprocessingError,
    InferenceError,
    PostprocessingError,
    CacheError,
)

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    # Exceptions
    "ArabicSummarizerError",
    "InvalidInputError",
    "InvalidRatioError",
    "UnsupportedLanguageError",
    "ModelNotFoundError",
    "ModelLoadError",
    "PreprocessingError",
    "InferenceError",
    "PostprocessingError",
    "CacheError",
]