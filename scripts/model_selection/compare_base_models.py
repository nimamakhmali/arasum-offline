"""
TASK-102: مقایسه عملی مدل‌های پایه برای انتخاب مدل مناسب Fine-tuning.

این اسکریپت:
- مدل‌های کاندید را از config می‌خواند
- روی نمونه‌های یکسان inference انجام می‌دهد
- زمان و کیفیت هر مدل را اندازه‌گیری می‌کند
- نتایج را ذخیره می‌کند
- یک مدل پیشنهادی برای Fine-tuning ارائه می‌دهد

اجرا:
    python scripts/model_selection/compare_base_models.py
    python scripts/model_selection/compare_base_models.py --num-samples 20 --device cpu
    python scripts/model_selection/compare_base_models.py --models arabart,arat5
    python scripts/model_selection/compare_base_models.py --dataset-path data/processed/xlsum_arabic/validation.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import re
# اضافه کردن src به path برای import
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from arabic_summarizer.evaluation.rouge_scorer import ArabicRougeScorer, RougeScores
from arabic_summarizer.utils.config_loader import ConfigLoader
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


# Data Structures



@dataclass
class ModelConfig:
    """تنظیمات یک مدل کاندید."""

    model_id: str
    model_name: str
    architecture: str
    model_type: str
    tokenizer_id: str
    max_input_length: int
    enabled: bool
    priority: int
    arabic_optimized: bool
    task_prefix: str | None
    notes: str
    generation_params_override: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """نتیجه inference روی یک نمونه."""

    sample_id: str
    input_text: str
    reference_summary: str | None
    generated_summary: str
    inference_time_seconds: float
    preprocessing_time_seconds: float
    rouge_scores: RougeScores | None
    status: str  # "success" | "failed"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "input_text": self.input_text[:500] + "..."
            if len(self.input_text) > 500
            else self.input_text,
            "reference_summary": self.reference_summary,
            "generated_summary": self.generated_summary,
            "inference_time_seconds": round(self.inference_time_seconds, 4),
            "preprocessing_time_seconds": round(self.preprocessing_time_seconds, 4),
            "rouge_scores": self.rouge_scores.to_dict()
            if self.rouge_scores
            else None,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class ModelBenchmarkResult:
    """نتیجه کامل benchmark یک مدل."""

    model_id: str
    model_name: str
    architecture: str
    status: str  # "success" | "failed" | "partial"
    error_message: str | None
    load_time_seconds: float
    sample_results: list[SampleResult] = field(default_factory=list)
    parameter_count: int | None = None

    @property
    def successful_samples(self) -> list[SampleResult]:
        return [s for s in self.sample_results if s.status == "success"]

    @property
    def failed_samples(self) -> list[SampleResult]:
        return [s for s in self.sample_results if s.status == "failed"]

    @property
    def aggregate_rouge(self) -> RougeScores | None:
        """میانگین ROUGE روی نمونه‌های موفق با reference."""
        scored = [
            s.rouge_scores
            for s in self.successful_samples
            if s.rouge_scores is not None
        ]
        if not scored:
            return None
        return RougeScores.average(scored)

    @property
    def inference_times(self) -> list[float]:
        return [s.inference_time_seconds for s in self.successful_samples]

    @property
    def avg_inference_time(self) -> float | None:
        times = self.inference_times
        return statistics.mean(times) if times else None

    @property
    def median_inference_time(self) -> float | None:
        times = self.inference_times
        return statistics.median(times) if times else None

    @property
    def total_inference_time(self) -> float:
        return sum(self.inference_times)

    def to_summary_dict(self) -> dict[str, Any]:
        rouge = self.aggregate_rouge
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "status": self.status,
            "error_message": self.error_message,
            "metrics": rouge.to_dict() if rouge else {},
            "performance": {
                "load_time_seconds": round(self.load_time_seconds, 4),
                "total_inference_time_seconds": round(self.total_inference_time, 4),
                "average_inference_time_seconds": round(self.avg_inference_time, 4)
                if self.avg_inference_time is not None
                else None,
                "median_inference_time_seconds": round(self.median_inference_time, 4)
                if self.median_inference_time is not None
                else None,
            },
            "samples": {
                "total": len(self.sample_results),
                "successful": len(self.successful_samples),
                "failed": len(self.failed_samples),
            },
            "parameter_count": self.parameter_count,
        }



# Dataset Loader



def load_benchmark_samples(
    dataset_path: Path | None,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    نمونه‌های benchmark را با determinism کامل بارگذاری می‌کند.

    اولویت:
    1. مسیر مشخص‌شده توسط کاربر
    2. فایل validation آماده‌شده پروژه
    3. فایل test آماده‌شده پروژه
    4. sample texts موجود در resources

    Args:
        dataset_path: مسیر دستی به فایل JSONL
        num_samples: تعداد نمونه‌های benchmark
        seed: seed برای انتخاب تصادفی deterministic

    Returns:
        لیست دیکشنری‌ها با کلیدهای id، text، summary
    """
    candidates: list[Path] = []

    if dataset_path:
        candidates.append(dataset_path)

    # مسیرهای پیش‌فرض پروژه
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    for source_dir in processed_dir.iterdir() if processed_dir.exists() else []:
        if source_dir.is_dir():
            for split in ["validation.jsonl", "test.jsonl", "train.jsonl"]:
                p = source_dir / split
                if p.exists():
                    candidates.append(p)

    records: list[dict[str, Any]] = []

    for path in candidates:
        if not path.exists():
            logger.warning("فایل داده پیدا نشد: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = rec.get("text", "").strip()
                    summary = rec.get("summary", "").strip()
                    if text:
                        records.append(
                            {
                                "id": rec.get(
                                    "id",
                                    rec.get(
                                        "text_hash",
                                        f"sample_{len(records)}",
                                    ),
                                ),
                                "text": text,
                                "summary": summary if summary else None,
                            }
                        )
            if records:
                logger.info(
                    "%d رکورد از %s بارگذاری شد.", len(records), path.name
                )
                break
        except Exception as exc:
            logger.warning("خطا در خواندن %s: %s", path, exc)
            continue

    if not records:
        logger.error(
            "هیچ داده‌ای برای benchmark پیدا نشد. "
            "لطفاً ابتدا scripts/data/download_datasets.py را اجرا کنید."
        )
        return []

    # انتخاب deterministic
    rng = random.Random(seed)
    if len(records) > num_samples:
        records = rng.sample(records, num_samples)
    else:
        logger.warning(
            "تعداد رکوردهای موجود (%d) کمتر از num_samples (%d) است.",
            len(records),
            num_samples,
        )

    logger.info(
        "%d نمونه برای benchmark انتخاب شد (seed=%d).", len(records), seed
    )
    return records



# Model Runner


class ModelRunner:
    """
    بارگذاری و اجرای inference یک مدل Seq2Seq.

    اصلاحات نسبت به نسخه قبل:
    - پشتیبانی از forced_bos_token_id برای mBART
    - تشخیص خودکار خروجی‌های معیوب (مثل <extra_id_X>)
    - لاگ بهتر برای debugging
    """

    # الگوی خروجی معیوب mT5/T5 بدون fine-tuning
    _GARBAGE_OUTPUT_PATTERN = re.compile(
        r"^(\s*<extra_id_\d+>\s*){2,}",
        re.MULTILINE,
    )

    def __init__(
        self,
        model_cfg: ModelConfig,
        device: str,
        base_generation_params: dict[str, Any],
    ) -> None:
        self._cfg = model_cfg
        self._device = device
        self._base_gen_params = {**base_generation_params}
        self._base_gen_params.update(model_cfg.generation_params_override)
        self._model = None
        self._tokenizer = None

    def load(self) -> float:
        """مدل و tokenizer را بارگذاری می‌کند."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(
            "بارگذاری مدل: %s از %s ...",
            self._cfg.model_name,
            self._cfg.model_id,
        )
        t0 = time.perf_counter()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._cfg.tokenizer_id,
                use_fast=True,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self._cfg.model_id,
            )
            self._model.eval()
            self._model.to(self._device)

            load_time = time.perf_counter() - t0
            param_count = sum(p.numel() for p in self._model.parameters())
            logger.info(
                "مدل %s بارگذاری شد. پارامترها: %s | زمان: %.2fs",
                self._cfg.model_name,
                f"{param_count:,}",
                load_time,
            )
            return load_time

        except Exception as exc:
            raise RuntimeError(
                f"بارگذاری مدل {self._cfg.model_id} ناموفق بود: {exc}"
            ) from exc

    def get_parameter_count(self) -> int | None:
        if self._model is None:
            return None
        return sum(p.numel() for p in self._model.parameters())

    def _is_garbage_output(self, text: str) -> bool:
        """
        بررسی می‌کند که خروجی مدل معیوب است یا نه.

        مدل‌های T5/mT5 بدون fine-tuning روی summarization
        ممکن است خروجی‌هایی مثل:
            <extra_id_0> ... <extra_id_10> ...
        تولید کنند که هیچ معنایی ندارند.
        """
        if not text or not text.strip():
            return True
        if self._GARBAGE_OUTPUT_PATTERN.search(text):
            return True
        # اگر بیش از ۵۰٪ توکن‌های خروجی special token باشند
        special_token_count = text.count("<extra_id_")
        total_words = len(text.split())
        if total_words > 0 and special_token_count / total_words > 0.3:
            return True
        return False

    def _build_generation_kwargs(self, input_ids) -> dict[str, Any]:
        """
        پارامترهای generation را آماده می‌کند.

        برای mBART: forced_bos_token_id از config خوانده می‌شود.
        اگر در generation_params_override موجود باشد، از آن استفاده می‌شود.
        """
        kwargs = {**self._base_gen_params}

        # اگر forced_bos_token_id در params بود، آن را اعمال کن
        # (برای mBART که نیاز به تعیین زبان هدف دارد)
        if "forced_bos_token_id" not in kwargs:
            # تلاش برای تشخیص خودکار از tokenizer
            if (
                hasattr(self._tokenizer, "lang_code_to_id")
                and "ar_AR" in self._tokenizer.lang_code_to_id
            ):
                kwargs["forced_bos_token_id"] = (
                    self._tokenizer.lang_code_to_id["ar_AR"]
                )
                logger.debug(
                    "forced_bos_token_id برای mBART تنظیم شد: %d",
                    kwargs["forced_bos_token_id"],
                )

        return kwargs

    def generate(self, text: str) -> tuple[str, float, float]:
        """
        خلاصه متن ورودی را تولید می‌کند.

        Returns:
            (خلاصه تولیدشده، زمان preprocessing، زمان inference)

        Raises:
            RuntimeError: اگر مدل بارگذاری نشده باشد
            ValueError: اگر خروجی معیوب باشد
        """
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "مدل بارگذاری نشده. ابتدا load() را فراخوانی کنید."
            )

        # ── Preprocessing ─────────────────────────────────────
        t_pre = time.perf_counter()
        input_text = text
        if self._cfg.task_prefix:
            input_text = self._cfg.task_prefix + text

        inputs = self._tokenizer(
            input_text,
            max_length=self._cfg.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        preprocessing_time = time.perf_counter() - t_pre

        # ── Inference ─────────────────────────────────────────
        if self._device == "cuda":
            torch.cuda.synchronize()

        t_inf = time.perf_counter()
        gen_kwargs = self._build_generation_kwargs(inputs.get("input_ids"))

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        if self._device == "cuda":
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - t_inf

        # ── Decode ────────────────────────────────────────────
        summary = self._tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        # بررسی کیفیت خروجی
        if self._is_garbage_output(summary):
            logger.warning(
                "مدل %s خروجی معیوب تولید کرد: '%s...'",
                self._cfg.model_name,
                summary[:80],
            )
            raise ValueError(
                f"خروجی مدل {self._cfg.model_name} معیوب است "
                f"(احتمالاً نیاز به fine-tuning دارد): {summary[:80]}"
            )

        return summary, preprocessing_time, inference_time

    def unload(self) -> None:
        """مدل را از حافظه آزاد می‌کند."""
        import gc
        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("مدل %s از حافظه آزاد شد.", self._cfg.model_name)

# Benchmark Runner



class BenchmarkRunner:
    """
    اجرای کامل benchmark روی همه مدل‌های کاندید.

    این کلاس:
    - مدل‌ها را یکی یکی بارگذاری و اجرا می‌کند
    - در صورت خطای یک مدل، benchmark ادامه می‌یابد
    - نتایج را ذخیره می‌کند
    - مدل پیشنهادی را انتخاب می‌کند
    """

    def __init__(
        self,
        model_configs: list[ModelConfig],
        samples: list[dict[str, Any]],
        device: str,
        generation_params: dict[str, Any],
        ranking_weights: dict[str, float],
        output_dir: Path,
        save_model_outputs: bool = True,
    ) -> None:
        self._model_configs = model_configs
        self._samples = samples
        self._device = device
        self._generation_params = generation_params
        self._ranking_weights = ranking_weights
        self._output_dir = output_dir
        self._save_model_outputs = save_model_outputs
        self._rouge_scorer = ArabicRougeScorer()
        self._results: list[ModelBenchmarkResult] = []

    def run(self) -> list[ModelBenchmarkResult]:
        """
        Benchmark را روی همه مدل‌های فعال اجرا می‌کند.

        Returns:
            لیست نتایج برای هر مدل
        """
        enabled = [m for m in self._model_configs if m.enabled]
        logger.info(
            "Benchmark شروع شد. تعداد مدل: %d | نمونه: %d | device: %s",
            len(enabled),
            len(self._samples),
            self._device,
        )

        for model_cfg in sorted(enabled, key=lambda m: m.priority):
            result = self._run_single_model(model_cfg)
            self._results.append(result)

            if self._save_model_outputs and result.sample_results:
                self._save_sample_outputs(result)

        return self._results

    def _run_single_model(self, model_cfg: ModelConfig) -> ModelBenchmarkResult:
        """Benchmark یک مدل را اجرا می‌کند."""
        logger.info("── شروع benchmark مدل: %s ──", model_cfg.model_name)

        runner = ModelRunner(
            model_cfg=model_cfg,
            device=self._device,
            base_generation_params=self._generation_params,
        )

        # ── بارگذاری مدل ───
        try:
            load_time = runner.load()
        except RuntimeError as exc:
            logger.error("بارگذاری مدل %s ناموفق: %s", model_cfg.model_name, exc)
            return ModelBenchmarkResult(
                model_id=model_cfg.model_id,
                model_name=model_cfg.model_name,
                architecture=model_cfg.architecture,
                status="failed",
                error_message=str(exc),
                load_time_seconds=0.0,
            )

        param_count = runner.get_parameter_count()
        sample_results: list[SampleResult] = []

        # ── inference روی هر نمونه ───────────────────────────
        for sample in self._samples:
            sample_result = self._run_single_sample(
                runner=runner,
                sample=sample,
                model_name=model_cfg.model_name,
            )
            sample_results.append(sample_result)

        runner.unload()

        # تعیین وضعیت کلی
        n_success = sum(1 for s in sample_results if s.status == "success")
        if n_success == 0:
            status = "failed"
        elif n_success < len(sample_results):
            status = "partial"
        else:
            status = "success"

        result = ModelBenchmarkResult(
            model_id=model_cfg.model_id,
            model_name=model_cfg.model_name,
            architecture=model_cfg.architecture,
            status=status,
            error_message=None,
            load_time_seconds=load_time,
            sample_results=sample_results,
            parameter_count=param_count,
        )

        rouge = result.aggregate_rouge
        if rouge:
            logger.info(
                "نتایج %s: ROUGE-1=%.3f | ROUGE-2=%.3f | ROUGE-L=%.3f | "
                "avg_time=%.2fs | موفق=%d/%d",
                model_cfg.model_name,
                rouge.rouge1,
                rouge.rouge2,
                rouge.rougeL,
                result.avg_inference_time or 0,
                n_success,
                len(sample_results),
            )
        else:
            logger.info(
                "نتایج %s: reference summary موجود نبود | "
                "avg_time=%.2fs | موفق=%d/%d",
                model_cfg.model_name,
                result.avg_inference_time or 0,
                n_success,
                len(sample_results),
            )

        return result

    def _run_single_sample(
        self,
        runner: ModelRunner,
        sample: dict[str, Any],
        model_name: str,
    ) -> SampleResult:
        """inference روی یک نمونه را اجرا می‌کند."""
        sample_id = str(sample.get("id", "unknown"))
        text = sample["text"]
        reference = sample.get("summary")

        try:
            generated, pre_time, inf_time = runner.generate(text)

            rouge_scores = None
            if reference and generated:
                rouge_scores = self._rouge_scorer.score(generated, reference)

            return SampleResult(
                sample_id=sample_id,
                input_text=text,
                reference_summary=reference,
                generated_summary=generated,
                inference_time_seconds=inf_time,
                preprocessing_time_seconds=pre_time,
                rouge_scores=rouge_scores,
                status="success",
            )

        except Exception as exc:
            logger.warning(
                "خطا در inference مدل %s روی نمونه %s: %s",
                model_name,
                sample_id,
                exc,
            )
            return SampleResult(
                sample_id=sample_id,
                input_text=text,
                reference_summary=reference,
                generated_summary="",
                inference_time_seconds=0.0,
                preprocessing_time_seconds=0.0,
                rouge_scores=None,
                status="failed",
                error_message=str(exc),
            )

    def _save_sample_outputs(self, result: ModelBenchmarkResult) -> None:
        """خروجی نمونه‌ها را برای یک مدل ذخیره می‌کند."""
        outputs_dir = self._output_dir / "model_outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        safe_name = result.model_name.replace("/", "_").replace(" ", "_")
        out_file = outputs_dir / f"{safe_name}.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            for s in result.sample_results:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        logger.debug("خروجی نمونه‌ها ذخیره شد: %s", out_file)



# Ranking
def compute_ranking_score(
    result: ModelBenchmarkResult,
    weights: dict[str, float],
    all_results: list[ModelBenchmarkResult],
) -> float:
    """
    امتیاز ranking یک مدل را محاسبه می‌کند.

    معیارها:
    - کیفیت ROUGE (rouge1، rouge2، rougeL)
    - سرعت (inverse normalized inference time)

    نرمال‌سازی نسبی بین مدل‌ها انجام می‌شود تا مقایسه منصفانه باشد.
    اگر تنها یک مدل موفق باشد، نرمال‌سازی روی همان انجام می‌شود.

    Args:
        result: نتیجه مدل مورد نظر
        weights: وزن‌های هر معیار از config
        all_results: نتایج همه مدل‌ها برای نرمال‌سازی

    Returns:
        امتیاز نهایی بین 0 و 1
    """
    if result.status == "failed":
        return 0.0

    rouge = result.aggregate_rouge
    if rouge is None:
        rouge = RougeScores()

    # جمع‌آوری مقادیر برای نرمال‌سازی
    successful = [r for r in all_results if r.status != "failed"]

    def _normalize(value: float, values: list[float]) -> float:
        if not values or max(values) == min(values):
            return 1.0
        return (value - min(values)) / (max(values) - min(values))

    all_rouge1 = [r.aggregate_rouge.rouge1 if r.aggregate_rouge else 0 for r in successful]
    all_rouge2 = [r.aggregate_rouge.rouge2 if r.aggregate_rouge else 0 for r in successful]
    all_rougeL = [r.aggregate_rouge.rougeL if r.aggregate_rouge else 0 for r in successful]
    all_times = [r.avg_inference_time or float("inf") for r in successful]

    norm_rouge1 = _normalize(rouge.rouge1, all_rouge1)
    norm_rouge2 = _normalize(rouge.rouge2, all_rouge2)
    norm_rougeL = _normalize(rouge.rougeL, all_rougeL)

    # برای سرعت: کمتر بهتر است، پس معکوس نرمال می‌کنیم
    avg_time = result.avg_inference_time or float("inf")
    if all_times and max(all_times) != min(all_times):
        norm_speed = 1.0 - (avg_time - min(all_times)) / (max(all_times) - min(all_times))
    else:
        norm_speed = 1.0

    w_r1 = weights.get("rouge1", 0.25)
    w_r2 = weights.get("rouge2", 0.35)
    w_rL = weights.get("rougeL", 0.30)
    w_spd = weights.get("speed", 0.10)

    total_w = w_r1 + w_r2 + w_rL + w_spd
    if total_w == 0:
        return 0.0

    score = (
        w_r1 * norm_rouge1
        + w_r2 * norm_rouge2
        + w_rL * norm_rougeL
        + w_spd * norm_speed
    ) / total_w

    return round(score, 6)


def select_recommended_model(
    results: list[ModelBenchmarkResult],
    ranking_weights: dict[str, float],
) -> dict[str, str] | None:
    """
    مدل پیشنهادی را بر اساس ranking انتخاب می‌کند.

    Returns:
        دیکشنری شامل model_id، model_name، score، reason
        یا None اگر هیچ مدلی موفق نبود
    """
    successful = [r for r in results if r.status != "failed"]
    if not successful:
        logger.error("هیچ مدلی با موفقیت اجرا نشد.")
        return None

    scores: list[tuple[float, ModelBenchmarkResult]] = []
    for result in successful:
        score = compute_ranking_score(result, ranking_weights, results)
        scores.append((score, result))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_result = scores[0]

    rouge = best_result.aggregate_rouge
    rouge_info = (
        f"ROUGE-1={rouge.rouge1:.3f}, ROUGE-2={rouge.rouge2:.3f}, "
        f"ROUGE-L={rouge.rougeL:.3f}"
        if rouge
        else "ROUGE محاسبه نشد (reference موجود نبود)"
    )

    reason = (
        f"بالاترین امتیاز ranking (score={best_score:.4f}) با "
        f"{rouge_info} و "
        f"میانگین زمان inference={best_result.avg_inference_time:.2f}s. "
        f"وزن‌های استفاده‌شده: rouge1={ranking_weights.get('rouge1', 0)}, "
        f"rouge2={ranking_weights.get('rouge2', 0)}, "
        f"rougeL={ranking_weights.get('rougeL', 0)}, "
        f"speed={ranking_weights.get('speed', 0)}."
    )

    return {
        "model_id": best_result.model_id,
        "model_name": best_result.model_name,
        "ranking_score": best_score,
        "reason": reason,
        "all_rankings": [
            {
                "model_name": r.model_name,
                "model_id": r.model_id,
                "ranking_score": s,
            }
            for s, r in scores
        ],
    }



# Output Writers

def save_json_report(
    results: list[ModelBenchmarkResult],
    recommended: dict | None,
    metadata: dict[str, Any],
    output_dir: Path,
) -> Path:
    """گزارش JSON کامل را ذخیره می‌کند."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model_comparison_results.json"

    report = {
        "benchmark_metadata": metadata,
        "models": [r.to_summary_dict() for r in results],
        "recommended_model": recommended,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("گزارش JSON ذخیره شد: %s", out_path)
    return out_path


def save_csv_summary(
    results: list[ModelBenchmarkResult],
    output_dir: Path,
) -> Path:
    """خلاصه CSV نتایج را ذخیره می‌کند."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model_comparison_summary.csv"

    fieldnames = [
        "model_id",
        "model_name",
        "architecture",
        "status",
        "rouge1",
        "rouge2",
        "rougeL",
        "avg_inference_time",
        "median_inference_time",
        "load_time",
        "successful_samples",
        "failed_samples",
        "parameter_count",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            rouge = result.aggregate_rouge
            writer.writerow(
                {
                    "model_id": result.model_id,
                    "model_name": result.model_name,
                    "architecture": result.architecture,
                    "status": result.status,
                    "rouge1": round(rouge.rouge1, 4) if rouge else "",
                    "rouge2": round(rouge.rouge2, 4) if rouge else "",
                    "rougeL": round(rouge.rougeL, 4) if rouge else "",
                    "avg_inference_time": round(result.avg_inference_time, 4)
                    if result.avg_inference_time is not None
                    else "",
                    "median_inference_time": round(result.median_inference_time, 4)
                    if result.median_inference_time is not None
                    else "",
                    "load_time": round(result.load_time_seconds, 4),
                    "successful_samples": len(result.successful_samples),
                    "failed_samples": len(result.failed_samples),
                    "parameter_count": result.parameter_count or "",
                }
            )

    logger.info("خلاصه CSV ذخیره شد: %s", out_path)
    return out_path



# Config Reader

def load_model_configs_from_config(
                                    enabled_filter: list[str] | None = None,
                                    ) -> tuple[list[ModelConfig], dict[str, Any], dict[str, float], dict[str, Any]]:
    """
    تنظیمات مدل‌ها و benchmark را از model_config.yaml می‌خواند.

    Args:
        enabled_filter: اگر مشخص شده، فقط مدل‌هایی با این نام‌ها برگردانده می‌شوند

    Returns:
        (لیست ModelConfig، generation_params، ranking_weights، benchmark_cfg)
    """
    cfg = ConfigLoader()

    benchmark_cfg: dict[str, Any] = cfg.get_section("model_config", "benchmark")
    if not benchmark_cfg:
        raise ValueError(
            "بخش 'benchmark' در model_config.yaml پیدا نشد. "
            "لطفاً فایل config را بررسی کنید."
        )

    generation_params: dict[str, Any] = benchmark_cfg.get("generation_params", {})
    ranking_weights: dict[str, float] = benchmark_cfg.get("ranking_weights", {})
    raw_models: list[dict] = benchmark_cfg.get("candidate_models", [])

    if not raw_models:
        raise ValueError("هیچ مدل کاندیدی در model_config.yaml تعریف نشده.")

    model_configs: list[ModelConfig] = []
    for raw in raw_models:
        mc = ModelConfig(
            model_id=raw["model_id"],
            model_name=raw["model_name"],
            architecture=raw.get("architecture", "unknown"),
            model_type=raw.get("model_type", "seq2seq"),
            tokenizer_id=raw.get("tokenizer_id", raw["model_id"]),
            max_input_length=raw.get("max_input_length", 512),
            enabled=raw.get("enabled", True),
            priority=raw.get("priority", 99),
            arabic_optimized=raw.get("arabic_optimized", False),
            task_prefix=raw.get("task_prefix"),
            notes=raw.get("notes", ""),
            generation_params_override=raw.get("generation_params_override", {}),
        )
        if enabled_filter:
            name_lower = mc.model_name.lower()
            if not any(f.lower() in name_lower for f in enabled_filter):
                mc.enabled = False
        model_configs.append(mc)

    return model_configs, generation_params, ranking_weights, benchmark_cfg



# CLI



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TASK-102: مقایسه مدل‌های پایه خلاصه‌سازی عربی",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="تعداد نمونه‌های benchmark (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="device اجرا (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="نام مدل‌های مورد نظر با کاما جدا شده، مثلاً: arabart,arat5",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="مسیر فایل JSONL داده benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="پوشه خروجی (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed برای انتخاب تصادفی نمونه‌ها (پیش‌فرض از config)",
    )
    return parser.parse_args()


def validate_device(device: str) -> str:
    """device را اعتبارسنجی می‌کند و در صورت عدم موجودیت CUDA خطا می‌دهد."""
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA درخواست شده اما موجود نیست. "
                "از --device cpu استفاده کنید یا driver CUDA را نصب کنید."
            )
    return device


def main() -> None:
    args = parse_args()

    #  خواندن config 
    enabled_filter = args.models.split(",") if args.models else None
    try:
        model_configs, generation_params, ranking_weights, benchmark_cfg = (
            load_model_configs_from_config(enabled_filter)
        )
    except ValueError as exc:
        logger.error("خطای config: %s", exc)
        sys.exit(1)

    #  اعمال override های CLI 
    num_samples: int = args.num_samples or benchmark_cfg.get("num_samples", 20)
    seed: int = args.seed or benchmark_cfg.get("seed", 42)
    device_str: str = args.device or benchmark_cfg.get("device", "cpu")
    output_dir: Path = args.output_dir or _PROJECT_ROOT / benchmark_cfg.get(
        "output_dir", "reports"
    )
    save_outputs: bool = benchmark_cfg.get("save_model_outputs", True)

    try:
        device = validate_device(device_str)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    #  بارگذاری نمونه‌ها 
    samples = load_benchmark_samples(
        dataset_path=args.dataset_path,
        num_samples=num_samples,
        seed=seed,
    )

    if not samples:
        logger.error(
            "داده‌ای برای benchmark موجود نیست. "
            "ابتدا scripts/data/download_datasets.py را اجرا کنید."
        )
        sys.exit(1)

    #  اجرای benchmark 
    runner = BenchmarkRunner(
        model_configs=model_configs,
        samples=samples,
        device=device,
        generation_params=generation_params,
        ranking_weights=ranking_weights,
        output_dir=output_dir,
        save_model_outputs=save_outputs,
    )

    start_time = datetime.now(timezone.utc)
    results = runner.run()

    #  انتخاب مدل پیشنهادی 
    recommended = select_recommended_model(results, ranking_weights)

    #  ذخیره نتایج 
    metadata = {
        "timestamp": start_time.isoformat(),
        "device": device,
        "num_samples_requested": num_samples,
        "num_samples_actual": len(samples),
        "seed": seed,
        "dataset_source": str(args.dataset_path) if args.dataset_path else "auto",
        "generation_config": generation_params,
        "ranking_weights": ranking_weights,
    }

    json_path = save_json_report(results, recommended, metadata, output_dir)
    csv_path = save_csv_summary(results, output_dir)

    # ── نمایش خلاصه ────────
    logger.info("═══ خلاصه نتایج Benchmark ═══")
    for result in results:
        rouge = result.aggregate_rouge
        logger.info(
            "  %s [%s]: rouge1=%.3f rouge2=%.3f rougeL=%.3f avg_time=%.2fs",
            result.model_name,
            result.status,
            rouge.rouge1 if rouge else 0,
            rouge.rouge2 if rouge else 0,
            rouge.rougeL if rouge else 0,
            result.avg_inference_time or 0,
        )

    if recommended:
        logger.info(
            "═══ مدل پیشنهادی: %s (score=%.4f) ═══",
            recommended["model_name"],
            recommended["ranking_score"],
        )
        logger.info("دلیل: %s", recommended["reason"])

    logger.info("نتایج در %s و %s ذخیره شدند.", json_path, csv_path)


if __name__ == "__main__":
    main()