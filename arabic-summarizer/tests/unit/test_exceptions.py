"""
تست‌های واحد برای exceptions سفارشی.

اجرا:
    pytest tests/unit/test_exceptions.py -v
"""

from __future__ import annotations

import pytest

from arabic_summarizer.exceptions import (
    ArabicSummarizerError,
    CacheError,
    InferenceError,
    InvalidInputError,
    InvalidRatioError,
    ModelLoadError,
    ModelNotFoundError,
    PostprocessingError,
    PreprocessingError,
    UnsupportedLanguageError,
)


class TestExceptionHierarchy:
    """تست سلسله‌مراتب ارث‌بری exceptions."""

    def test_all_exceptions_inherit_from_base(self):
        """همه exceptions باید از ArabicSummarizerError ارث ببرند."""
        child_exceptions = [
            InvalidInputError,
            InvalidRatioError,
            UnsupportedLanguageError,
            ModelNotFoundError,
            ModelLoadError,
            PreprocessingError,
            InferenceError,
            PostprocessingError,
            CacheError,
        ]
        for exc_class in child_exceptions:
            assert issubclass(exc_class, ArabicSummarizerError), (
                f"{exc_class.__name__} باید از ArabicSummarizerError ارث ببرد"
            )

    def test_base_exception_inherits_from_exception(self):
        """ArabicSummarizerError باید از Exception ارث ببرد."""
        assert issubclass(ArabicSummarizerError, Exception)

    def test_catch_all_with_base_class(self):
        """باید بتوان همه خطاها را با کلاس پایه گرفت."""
        exceptions_to_test = [
            InvalidInputError(),
            ModelNotFoundError(),
            InferenceError(),
            PreprocessingError(),
        ]
        for exc in exceptions_to_test:
            with pytest.raises(ArabicSummarizerError):
                raise exc


class TestInvalidInputError:

    def test_default_message(self):
        """بدون argument باید پیام پیش‌فرض داشته باشد."""
        exc = InvalidInputError()
        assert str(exc) == InvalidInputError.default_message

    def test_with_word_count(self):
        """با word_count باید پیام توصیفی تولید کند."""
        exc = InvalidInputError(word_count=150)
        assert "150" in str(exc)
        assert "300" in str(exc)
        assert "3000" in str(exc)

    def test_word_count_attribute(self):
        """attribute های word_count، min_words، max_words باید ست شوند."""
        exc = InvalidInputError(word_count=50, min_words=300, max_words=3000)
        assert exc.word_count == 50
        assert exc.min_words == 300
        assert exc.max_words == 3000

    def test_custom_message_overrides_auto(self):
        """پیام سفارشی باید بر پیام خودکار غلبه کند."""
        exc = InvalidInputError(message="پیام سفارشی", word_count=100)
        assert str(exc) == "پیام سفارشی"

    def test_raises_correctly(self):
        """باید با pytest.raises قابل گرفتن باشد."""
        with pytest.raises(InvalidInputError):
            raise InvalidInputError(word_count=50)


class TestModelNotFoundError:

    def test_with_model_path(self):
        """با مسیر مدل باید مسیر را در پیام نشان دهد."""
        exc = ModelNotFoundError(model_path="/models/arabart/model.onnx")
        assert "/models/arabart/model.onnx" in str(exc)

    def test_model_path_attribute(self):
        """attribute model_path باید ست شود."""
        exc = ModelNotFoundError(model_path="/some/path")
        assert exc.model_path == "/some/path"

    def test_without_path(self):
        """بدون مسیر باید پیام پیش‌فرض داشته باشد."""
        exc = ModelNotFoundError()
        assert str(exc) == ModelNotFoundError.default_message


class TestInvalidRatioError:

    def test_with_ratio(self):
        """با ratio باید مقدار را در پیام نشان دهد."""
        exc = InvalidRatioError(ratio=0.5)
        assert "0.50" in str(exc)

    def test_ratio_attribute(self):
        """attribute ratio باید ست شود."""
        exc = InvalidRatioError(ratio=0.05)
        assert exc.ratio == pytest.approx(0.05)

    def test_default_message(self):
        """بدون ratio باید پیام پیش‌فرض داشته باشد."""
        exc = InvalidRatioError()
        assert str(exc) == InvalidRatioError.default_message


class TestUnsupportedLanguageError:

    def test_with_detected_language(self):
        """با زبان شناسایی‌شده باید آن را در پیام نشان دهد."""
        exc = UnsupportedLanguageError(detected_language="english")
        assert "english" in str(exc)

    def test_detected_language_attribute(self):
        """attribute detected_language باید ست شود."""
        exc = UnsupportedLanguageError(detected_language="farsi")
        assert exc.detected_language == "farsi"