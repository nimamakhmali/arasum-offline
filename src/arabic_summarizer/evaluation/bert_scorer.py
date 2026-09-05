"""
محاسبه BERTScore برای ارزیابی معنایی خلاصه‌سازی.

از bert-base-multilingual-cased استفاده می‌شود چون:
- از عربی پشتیبانی می‌کند
- آفلاین قابل اجراست (بعد از دانلود یک‌بار)
- هدف پروژه BERTScore F1 >= 0.85

نکته آفلاین:
    این ماژول در inference نیازی به اینترنت ندارد.
    اما مدل BERT باید از قبل دانلود شده باشد.
"""

from __future__ import annotations

from dataclasses import dataclass

from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BertScoreResult:
    """نتیجه BERTScore برای یک نمونه یا مجموعه."""

    precision: float
    recall: float
    f1: float
    model_used: str
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "model_used": self.model_used,
            "n_samples": self.n_samples,
            "meets_target": self.f1 >= 0.85,
        }


class ArabicBertScorer:
    """
    محاسبه BERTScore برای متون عربی.

    استفاده:
        scorer = ArabicBertScorer()
        result = scorer.score_batch(predictions, references)
        print(result.f1)
    """

    DEFAULT_MODEL = "bert-base-multilingual-cased"

    def __init__(self, model_type: str | None = None) -> None:
        self._model_type = model_type or self.DEFAULT_MODEL
        self._bert_score_available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import bert_score  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "bert-score نصب نیست. BERTScore غیرفعال می‌شود. "
                "برای نصب: pip install bert-score"
            )
            return False

    def score_batch(
        self,
        predictions: list[str],
        references: list[str],
        max_samples: int | None = None,
    ) -> BertScoreResult | None:
        """
        BERTScore را برای یک batch محاسبه می‌کند.

        Args:
            predictions: خلاصه‌های تولیدشده
            references: خلاصه‌های مرجع
            max_samples: حداکثر نمونه برای محاسبه (None = همه)

        Returns:
            BertScoreResult یا None در صورت خطا
        """
        if not self._bert_score_available:
            return None

        if len(predictions) != len(references):
            raise ValueError(
                f"طول predictions ({len(predictions)}) و "
                f"references ({len(references)}) باید برابر باشد."
            )

        if not predictions:
            return None

        # محدود کردن به max_samples
        if max_samples and len(predictions) > max_samples:
            import random
            rng = random.Random(42)
            indices = rng.sample(range(len(predictions)), max_samples)
            predictions = [predictions[i] for i in indices]
            references = [references[i] for i in indices]
            logger.info(
                "BERTScore روی %d نمونه محاسبه می‌شود (از %d کل).",
                max_samples,
                len(predictions) + (len(predictions) - max_samples),
            )

        try:
            from bert_score import score as bert_score_fn

            P, R, F1 = bert_score_fn(
                predictions,
                references,
                model_type=self._model_type,
                lang="ar",
                verbose=False,
                rescale_with_baseline=False,
            )

            result = BertScoreResult(
                precision=round(P.mean().item(), 4),
                recall=round(R.mean().item(), 4),
                f1=round(F1.mean().item(), 4),
                model_used=self._model_type,
                n_samples=len(predictions),
            )

            logger.info(
                "BERTScore: P=%.4f | R=%.4f | F1=%.4f (هدف ≥0.85: %s)",
                result.precision,
                result.recall,
                result.f1,
                "✓" if result.f1 >= 0.85 else "✗",
            )

            return result

        except Exception as exc:
            logger.error("خطا در محاسبه BERTScore: %s", exc)
            return None

    def is_available(self) -> bool:
        """آیا bert-score نصب است؟"""
        return self._bert_score_available