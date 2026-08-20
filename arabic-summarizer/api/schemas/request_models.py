"""
مدل‌های Pydantic برای اعتبارسنجی ورودی API.
"""

from pydantic import BaseModel, field_validator


class SummarizeRequest(BaseModel):
    text: str
    ratio: float = 0.2

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, value: str) -> str:
        word_count = len(value.split())
        if word_count < 300 or word_count > 3000:
            raise ValueError(
                f"طول متن باید بین 300 تا 3000 کلمه باشد. طول فعلی: {word_count}"
            )
        return value

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if not (0.1 <= value <= 0.3):
            raise ValueError("نسبت خلاصه‌سازی باید بین 0.1 تا 0.3 باشد.")
        return value
