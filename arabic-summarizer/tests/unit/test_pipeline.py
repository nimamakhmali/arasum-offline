"""
TODO: تست pipeline.run() با یک Mock Engine (بدون نیاز به مدل واقعی).
"""

from arabic_summarizer.pipeline import run


class MockEngine:
    def generate(self, text: str, ratio: float) -> str:
        return "این یک خلاصه فرضی است"


def test_pipeline_run_returns_formatted_summary():
    result = run("متن نرمال‌شده ورودی", ratio=0.2, engine=MockEngine())
    assert result.endswith(".")
