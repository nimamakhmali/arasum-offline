"""
اینترفیس پایه برای موتورهای استنتاج (Inference Engines).
هر Engine جدید (ONNX، احتمالا در آینده گزینه‌های دیگر) باید این کلاس را پیاده‌سازی کند.
"""

from abc import ABC, abstractmethod


class BaseEngine(ABC):

    @abstractmethod
    def load(self, model_path: str, tokenizer_path: str) -> None:
        """بارگذاری مدل و توکنایزر در حافظه"""
        raise NotImplementedError

    @abstractmethod
    def generate(self, text: str, ratio: float) -> str:
        """
        تولید خلاصه از متن ورودی.
        توکنایزیشن باید داخل همین متد انجام شود (encapsulation)،
        نه در سطح pipeline.
        """
        raise NotImplementedError
