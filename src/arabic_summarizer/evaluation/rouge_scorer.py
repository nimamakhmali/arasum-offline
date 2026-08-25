"""
محاسبه معیارهای ROUGE برای ارزیابی کیفیت خلاصه‌سازی.

این ماژول از کتابخانه rouge_score استفاده می‌کند و با
preprocessing موجود پروژه سازگار است.

نکته مهم: برای متن عربی از WhitespaceTokenizer استفاده می‌شود
چون tokenizer پیش‌فرض rouge_score برای عربی مناسب نیست.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rouge_score import rouge_scorer as rs

from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_arabic_for_rouge(text: str) -> str:
    """
    نرمال‌سازی حداقلی و قابل توضیح متن عربی برای ROUGE.

    این تابع فقط عملیاتی انجام می‌دهد که:
    1. deterministic باشد
    2. برای هر دو متن مرجع و خروجی یکسان اعمال شود
    3. معنای متن را تغییر ندهد

    از pipeline اصلی preprocessing پروژه استفاده نمی‌کند
    چون ROUGE باید روی متن summary اعمال شود نه متن ورودی.
    """
    if not text:
        return ""
    # حذف اعراب
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # حذف تطویل
    text = re.sub(r"\u0640", "", text)
    # یکسان‌سازی فاصله‌ها
    text = re.sub(r"\s+", " ", text).strip()
    return text


class _WhitespaceTokenizer:
    """
    Tokenizer مبتنی بر whitespace برای متون عربی.

    rouge_score با tokenizer پیش‌فرض خود (که برای انگلیسی طراحی شده)
    متن عربی را درست پردازش نمی‌کند زیرا regex های داخلی آن
    کاراکترهای غیر ASCII را drop می‌کند.

    این کلاس interface مورد انتظار rouge_score را پیاده‌سازی می‌کند
    و normalization عربی را قبل از tokenization اعمال می‌کند.
    """

    def tokenize(self, text: str) -> list[str]:
        """متن را بعد از normalization بر اساس whitespace tokenize می‌کند."""
        normalized = _normalize_arabic_for_rouge(text)
        return [t for t in normalized.split() if t]


@dataclass
class RougeScores:
    """نتیجه محاسبه ROUGE برای یک نمونه یا مجموعه."""

    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "rouge1": round(self.rouge1, 4),
            "rouge2": round(self.rouge2, 4),
            "rougeL": round(self.rougeL, 4),
        }

    @classmethod
    def average(cls, scores: list["RougeScores"]) -> "RougeScores":
        """میانگین چند نتیجه ROUGE را محاسبه می‌کند."""
        if not scores:
            return cls()
        n = len(scores)
        return cls(
            rouge1=sum(s.rouge1 for s in scores) / n,
            rouge2=sum(s.rouge2 for s in scores) / n,
            rougeL=sum(s.rougeL for s in scores) / n,
        )


class ArabicRougeScorer:
    """
    محاسبه ROUGE برای متون عربی.

    از _WhitespaceTokenizer استفاده می‌کند تا متن عربی
    به درستی tokenize شود.

    استفاده:
        scorer = ArabicRougeScorer()
        scores = scorer.score(hypothesis="خلاصه تولیدشده", reference="خلاصه مرجع")
    """

    def __init__(self) -> None:
        self._scorer = rs.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False,
            tokenizer=_WhitespaceTokenizer(),
        )

    def score(self, hypothesis: str, reference: str) -> RougeScores:
        """
        ROUGE را برای یک جفت (خروجی، مرجع) محاسبه می‌کند.

        Args:
            hypothesis: خلاصه تولیدشده توسط مدل
            reference: خلاصه مرجع انسانی

        Returns:
            RougeScores با مقادیر F1
        """
        if not hypothesis or not reference:
            logger.warning("متن خالی برای محاسبه ROUGE دریافت شد.")
            return RougeScores()

        # بررسی اینکه بعد از normalization توکن‌ای باقی می‌ماند
        hyp_tokens = _WhitespaceTokenizer().tokenize(hypothesis)
        ref_tokens = _WhitespaceTokenizer().tokenize(reference)

        if not hyp_tokens or not ref_tokens:
            logger.warning("بعد از normalization متن خالی شد.")
            return RougeScores()

        try:
            result = self._scorer.score(reference, hypothesis)
            return RougeScores(
                rouge1=result["rouge1"].fmeasure,
                rouge2=result["rouge2"].fmeasure,
                rougeL=result["rougeL"].fmeasure,
            )
        except Exception as exc:
            logger.error("خطا در محاسبه ROUGE: %s", exc)
            return RougeScores()

    def score_batch(
        self,
        hypotheses: list[str],
        references: list[str],
    ) -> tuple[list[RougeScores], RougeScores]:
        """
        ROUGE را برای یک batch محاسبه می‌کند.

        Returns:
            (لیست نتایج تکی، میانگین کل)
        """
        if len(hypotheses) != len(references):
            raise ValueError("طول hypotheses و references باید برابر باشد.")

        individual = [
            self.score(h, r) for h, r in zip(hypotheses, references)
        ]
        aggregate = RougeScores.average(individual)
        return individual, aggregate