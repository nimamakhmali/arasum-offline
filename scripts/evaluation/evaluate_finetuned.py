"""
TASK-203: ارزیابی جامع مدل Fine-Tuned AraBART.

این اسکریپت:
1. مدل Fine-Tuned را روی test set اجرا می‌کند
2. ROUGE-1/2/L را محاسبه می‌کند
3. BERTScore را محاسبه می‌کند
4. Latency baseline را اندازه می‌گیرد (TASK-203)
5. ارزیابی کیفی روی نمونه‌های نمونه‌وار
6. مقایسه مدل پایه vs Fine-Tuned

اجرا:
    python scripts/evaluation/evaluate_finetuned.py
    python scripts/evaluation/evaluate_finetuned.py --model-path models/finetuned/arabart/merged
    python scripts/evaluation/evaluate_finetuned.py --quick (فقط 100 نمونه)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from arabic_summarizer.evaluation.rouge_scorer import ArabicRougeScorer, RougeScores
from arabic_summarizer.utils.config_loader import ConfigLoader
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════

def load_model_for_eval(model_path: str, device: str) -> tuple[Any, Any]:
    """
    مدل Fine-Tuned را برای evaluation بارگذاری می‌کند.

    اولویت بارگذاری:
    1. مدل merge‌شده (merged/) - بهترین برای inference
    2. LoRA adapter (lora_adapter/) - اگر merge موجود نباشد
    3. مدل پایه (فقط برای مقایسه baseline)

    Args:
        model_path: مسیر مدل
        device: cpu یا cuda

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    path = Path(model_path)

    # بررسی وجود مدل merge‌شده
    merged_path = path / "merged"
    lora_path = path / "lora_adapter"

    if merged_path.exists():
        actual_path = str(merged_path)
        logger.info("بارگذاری مدل merge‌شده از: %s", actual_path)
    elif lora_path.exists():
        logger.info("بارگذاری LoRA adapter از: %s", lora_path)
        # بارگذاری با PEFT
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM
        cfg_loader = ConfigLoader()
        base_model_id = cfg_loader.get("training", "model.base_model_id", "moussaKam/AraBART")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
        model = PeftModel.from_pretrained(base_model, str(lora_path))
        model = model.merge_and_unload()
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
        logger.info("LoRA adapter بارگذاری و merge شد.")
        return model, tokenizer
    elif path.exists():
        actual_path = str(path)
        logger.info("بارگذاری مدل از: %s", actual_path)
    else:
        raise FileNotFoundError(
            f"مدل در مسیر {model_path} پیدا نشد. "
            "ابتدا finetune.py را اجرا کنید."
        )

    tokenizer = AutoTokenizer.from_pretrained(actual_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(actual_path)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "مدل بارگذاری شد: %s پارامتر | device: %s",
        f"{param_count:,}",
        device,
    )

    return model, tokenizer


# ══════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════

def generate_summary(
    model: Any,
    tokenizer: Any,
    text: str,
    device: str,
    gen_config: dict,
    max_source_length: int = 1024,
) -> tuple[str, float]:
    """
    خلاصه یک متن را تولید می‌کند.

    Returns:
        (خلاصه تولیدشده, زمان inference به ثانیه)
    """
    import torch

    inputs = tokenizer(
        text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=gen_config.get("num_beams", 4),
            early_stopping=gen_config.get("early_stopping", True),
            no_repeat_ngram_size=gen_config.get("no_repeat_ngram_size", 3),
            length_penalty=gen_config.get("length_penalty", 1.0),
            repetition_penalty=gen_config.get("repetition_penalty", 1.2),
            min_new_tokens=gen_config.get("min_new_tokens", 20),
            max_new_tokens=gen_config.get("max_new_tokens", 128),
        )

    if device == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - t_start

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return summary, inference_time


# ══════════════════════════════════════════════════════════
# ROUGE Evaluation
# ══════════════════════════════════════════════════════════

def evaluate_rouge(
    model: Any,
    tokenizer: Any,
    test_data: list[dict],
    device: str,
    gen_config: dict,
    max_source_length: int = 1024,
    max_samples: Optional[int] = None,
) -> tuple[dict, list[dict]]:
    """
    ROUGE را روی test set محاسبه می‌کند.

    نکته مهم: test set هیچ‌گاه برای tuning استفاده نشده
    و این اولین بار است که روی آن ارزیابی می‌شود.

    Returns:
        (آمار کلی, لیست نتایج تکی)
    """
    from tqdm import tqdm

    scorer = ArabicRougeScorer()

    if max_samples and max_samples < len(test_data):
        import random
        rng = random.Random(42)
        test_data = rng.sample(test_data, max_samples)
        logger.info("ارزیابی روی %d نمونه از test set", max_samples)

    results = []
    rouge_scores = []
    inference_times = []
    n_failed = 0

    for i, record in enumerate(tqdm(test_data, desc="ارزیابی ROUGE")):
        text = record.get("text", "")
        reference = record.get("summary", "")

        if not text or not reference:
            n_failed += 1
            continue

        try:
            prediction, inf_time = generate_summary(
                model, tokenizer, text, device, gen_config, max_source_length
            )

            if not prediction:
                n_failed += 1
                continue

            rouge = scorer.score(prediction, reference)
            rouge_scores.append(rouge)
            inference_times.append(inf_time)

            results.append({
                "id": record.get("id", str(i)),
                "input_words": len(text.split()),
                "reference_words": len(reference.split()),
                "prediction_words": len(prediction.split()),
                "compression_ratio": round(
                    len(prediction.split()) / len(text.split()), 4
                ) if text else 0,
                "rouge1": round(rouge.rouge1, 4),
                "rouge2": round(rouge.rouge2, 4),
                "rougeL": round(rouge.rougeL, 4),
                "inference_time": round(inf_time, 3),
                "reference": reference,
                "prediction": prediction,
            })

        except Exception as e:
            logger.warning("خطا در نمونه %d: %s", i, e)
            n_failed += 1

    if not rouge_scores:
        logger.error("هیچ نمونه‌ای با موفقیت ارزیابی نشد!")
        return {}, []

    aggregate = RougeScores.average(rouge_scores)

    stats = {
        "n_samples": len(results),
        "n_failed": n_failed,
        "rouge1": round(aggregate.rouge1, 4),
        "rouge2": round(aggregate.rouge2, 4),
        "rougeL": round(aggregate.rougeL, 4),
        "meets_rouge1_target": aggregate.rouge1 >= 0.45,
        "meets_rouge2_target": aggregate.rouge2 >= 0.25,
        "meets_rougeL_target": aggregate.rougeL >= 0.40,
        "inference_time": {
            "mean": round(statistics.mean(inference_times), 3),
            "median": round(statistics.median(inference_times), 3),
            "min": round(min(inference_times), 3),
            "max": round(max(inference_times), 3),
            "stdev": round(statistics.stdev(inference_times), 3) if len(inference_times) > 1 else 0,
        },
        "compression_ratio": {
            "mean": round(
                statistics.mean(r["compression_ratio"] for r in results), 4
            ) if results else 0,
        },
    }

    logger.info(
        "نتایج ROUGE:\n"
        "  ROUGE-1: %.4f (هدف ≥0.45: %s)\n"
        "  ROUGE-2: %.4f (هدف ≥0.25: %s)\n"
        "  ROUGE-L: %.4f (هدف ≥0.40: %s)\n"
        "  میانگین inference: %.3f ثانیه",
        aggregate.rouge1, "✓" if stats["meets_rouge1_target"] else "✗",
        aggregate.rouge2, "✓" if stats["meets_rouge2_target"] else "✗",
        aggregate.rougeL, "✓" if stats["meets_rougeL_target"] else "✗",
        stats["inference_time"]["mean"],
    )

    return stats, results


# ══════════════════════════════════════════════════════════
# BERTScore Evaluation
# ══════════════════════════════════════════════════════════

def evaluate_bertscore(
    predictions: list[str],
    references: list[str],
    bertscore_model: str = "bert-base-multilingual-cased",
    max_samples: int = 500,
) -> dict:
    """
    BERTScore را محاسبه می‌کند.

    از bert-base-multilingual-cased استفاده می‌شود چون:
    - از عربی پشتیبانی می‌کند
    - آفلاین قابل اجراست (بعد از دانلود یک‌بار)
    - معیار استاندارد پروژه است

    Args:
        predictions: خلاصه‌های تولیدشده
        references: خلاصه‌های مرجع
        bertscore_model: مدل BERTScore
        max_samples: حداکثر نمونه (BERTScore گران است)

    Returns:
        dict با precision, recall, f1
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.error("bert-score نصب نیست! pip install bert-score")
        return {"error": "bert-score not installed"}

    if len(predictions) > max_samples:
        logger.info(
            "BERTScore روی %d نمونه از %d اجرا می‌شود.",
            max_samples, len(predictions),
        )
        import random
        rng = random.Random(42)
        indices = rng.sample(range(len(predictions)), max_samples)
        predictions = [predictions[i] for i in indices]
        references = [references[i] for i in indices]

    logger.info(
        "محاسبه BERTScore با مدل '%s' روی %d نمونه ...",
        bertscore_model,
        len(predictions),
    )

    try:
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type=bertscore_model,
            lang="ar",
            verbose=False,
            rescale_with_baseline=False,
        )

        p_mean = round(P.mean().item(), 4)
        r_mean = round(R.mean().item(), 4)
        f1_mean = round(F1.mean().item(), 4)

        result = {
            "model": bertscore_model,
            "n_samples": len(predictions),
            "precision": p_mean,
            "recall": r_mean,
            "f1": f1_mean,
            "meets_target": f1_mean >= 0.85,
        }

        logger.info(
            "BERTScore:\n"
            "  Precision: %.4f\n"
            "  Recall:    %.4f\n"
            "  F1:        %.4f (هدف ≥0.85: %s)",
            p_mean, r_mean, f1_mean,
            "✓" if result["meets_target"] else "✗",
        )

        return result

    except Exception as e:
        logger.error("خطا در محاسبه BERTScore: %s", e)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
# Latency Benchmark (TASK-203)
# ══════════════════════════════════════════════════════════

def run_latency_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    gen_config: dict,
    test_data: list[dict],
    input_lengths: list[int],
    warmup_runs: int = 2,
    timed_runs: int = 5,
    max_source_length: int = 1024,
) -> dict:
    """
    Latency baseline را اندازه می‌گیرد.

    این baseline برای مقایسه با فاز ۳ (ONNX) استفاده می‌شود.
    Phase 2 requirement: < 60 ثانیه برای متن ≈1000 کلمه

    روش:
    - برای هر طول ورودی، نزدیک‌ترین نمونه از test_data انتخاب می‌شود
    - warmup_runs برای گرم کردن مدل
    - timed_runs برای اندازه‌گیری واقعی
    - model loading time حذف می‌شود

    Args:
        input_lengths: طول‌های مورد آزمایش (کلمه)
        warmup_runs: تعداد run های warmup
        timed_runs: تعداد run های زمان‌سنجی
    """
    import torch

    logger.info("══ شروع Latency Benchmark ══")

    # ایجاد نمونه‌های test به ازای هر طول
    def find_sample_for_length(target_length: int) -> Optional[str]:
        """نزدیک‌ترین نمونه به طول هدف را پیدا می‌کند."""
        if not test_data:
            return None
        best = min(
            test_data,
            key=lambda r: abs(len(r.get("text", "").split()) - target_length),
        )
        return best.get("text", "")

    benchmark_results = {}

    for target_length in input_lengths:
        sample_text = find_sample_for_length(target_length)
        if not sample_text:
            # ساخت متن مصنوعی
            sample_text = "أعلنت الحكومة عن برنامج وطني جديد لتطوير قطاع التعليم. " * (target_length // 10 + 1)
            sample_text = " ".join(sample_text.split()[:target_length])

        actual_words = len(sample_text.split())
        logger.info(
            "اندازه‌گیری latency برای ~%d کلمه (واقعی: %d) ...",
            target_length, actual_words,
        )

        # Warmup
        for _ in range(warmup_runs):
            try:
                generate_summary(model, tokenizer, sample_text, device, gen_config, max_source_length)
            except Exception:
                pass

        # Timed runs
        times = []
        output_lengths = []

        for _ in range(timed_runs):
            try:
                summary, t = generate_summary(
                    model, tokenizer, sample_text, device, gen_config, max_source_length
                )
                times.append(t)
                output_lengths.append(len(summary.split()))
            except Exception as e:
                logger.warning("خطا در benchmark: %s", e)

        if not times:
            benchmark_results[str(target_length)] = {"error": "همه run ها شکست خوردند"}
            continue

        result = {
            "target_input_words": target_length,
            "actual_input_words": actual_words,
            "warmup_runs": warmup_runs,
            "timed_runs": len(times),
            "mean_seconds": round(statistics.mean(times), 3),
            "median_seconds": round(statistics.median(times), 3),
            "min_seconds": round(min(times), 3),
            "max_seconds": round(max(times), 3),
            "stdev_seconds": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
            "avg_output_words": round(statistics.mean(output_lengths), 1) if output_lengths else 0,
            "device": device,
        }

        # Phase 2 requirement check برای ~1000 کلمه
        if 900 <= target_length <= 1100:
            result["phase2_requirement_60s"] = (
                "PASS" if result["mean_seconds"] < 60 else "FAIL"
            )
            logger.info(
                "Phase 2 requirement (<60s برای ~1000 کلمه): %s (%.2f ثانیه)",
                result["phase2_requirement_60s"],
                result["mean_seconds"],
            )

        benchmark_results[str(target_length)] = result

    return benchmark_results


# ══════════════════════════════════════════════════════════
# Qualitative Evaluation
# ══════════════════════════════════════════════════════════

def qualitative_evaluation(
    model: Any,
    tokenizer: Any,
    test_data: list[dict],
    device: str,
    gen_config: dict,
    n_samples: int = 10,
    max_source_length: int = 1024,
) -> list[dict]:
    """
    ارزیابی کیفی روی نمونه‌های منتخب.

    نمونه‌ها از محدوده‌های مختلف طول انتخاب می‌شوند.
    بررسی انسانی روی این نمونه‌ها لازم است.
    """
    if not test_data:
        return []

    # انتخاب نمونه‌های متنوع از محدوده‌های مختلف
    sorted_data = sorted(test_data, key=lambda r: len(r.get("text", "").split()))
    n = len(sorted_data)
    indices = [int(n * i / n_samples) for i in range(n_samples)]
    selected = [sorted_data[min(i, n-1)] for i in indices]

    qualitative_results = []
    scorer = ArabicRougeScorer()

    for i, record in enumerate(selected):
        text = record.get("text", "")
        reference = record.get("summary", "")

        if not text:
            continue

        try:
            prediction, inf_time = generate_summary(
                model, tokenizer, text, device, gen_config, max_source_length
            )

            rouge = scorer.score(prediction, reference) if reference else None

            qualitative_results.append({
                "sample_id": i + 1,
                "input_words": len(text.split()),
                "reference_words": len(reference.split()) if reference else 0,
                "prediction_words": len(prediction.split()),
                "input_preview": text[:200] + "..." if len(text) > 200 else text,
                "reference": reference,
                "prediction": prediction,
                "rouge1": round(rouge.rouge1, 4) if rouge else None,
                "rougeL": round(rouge.rougeL, 4) if rouge else None,
                "inference_time": round(inf_time, 3),
                "qualitative_notes": {
                    "factual_consistency": "TODO: بررسی انسانی لازم است",
                    "coherence": "TODO: بررسی انسانی لازم است",
                    "arabic_fluency": "TODO: بررسی انسانی لازم است",
                    "hallucination_detected": "TODO: بررسی انسانی لازم است",
                },
            })

        except Exception as e:
            logger.warning("خطا در ارزیابی کیفی نمونه %d: %s", i, e)

    return qualitative_results


# ══════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════

def generate_evaluation_report(
    rouge_stats: dict,
    bertscore_stats: dict,
    latency_stats: dict,
    qualitative_samples: list[dict],
    model_path: str,
    output_dir: Path,
) -> None:
    """
    گزارش ارزیابی کامل را ذخیره می‌کند.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── گزارش JSON ─────────────────────────────────────────
    full_report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "phase": "Phase 2 - Fine-Tuned AraBART Baseline",
        "rouge_evaluation": rouge_stats,
        "bertscore_evaluation": bertscore_stats,
        "latency_benchmark": latency_stats,
        "qualitative_samples_count": len(qualitative_samples),
        "project_targets": {
            "rouge1": {"target": 0.45, "achieved": rouge_stats.get("rouge1", 0),
                       "status": "PASS" if rouge_stats.get("meets_rouge1_target") else "FAIL"},
            "rouge2": {"target": 0.25, "achieved": rouge_stats.get("rouge2", 0),
                       "status": "PASS" if rouge_stats.get("meets_rouge2_target") else "FAIL"},
            "rougeL": {"target": 0.40, "achieved": rouge_stats.get("rougeL", 0),
                       "status": "PASS" if rouge_stats.get("meets_rougeL_target") else "FAIL"},
            "bertscore_f1": {
                "target": 0.85,
                "achieved": bertscore_stats.get("f1", 0),
                "status": "PASS" if bertscore_stats.get("meets_target") else "FAIL/UNKNOWN",
            },
            "latency_1000w_60s": {
                "target": "< 60 seconds",
                "achieved": latency_stats.get("1000", {}).get("mean_seconds", "N/A"),
                "status": latency_stats.get("1000", {}).get("phase2_requirement_60s", "UNKNOWN"),
            },
        },
        "known_limitations": [
            "XL-Sum reference summaries avg compression ≈5%, product target 10-30%",
            "AraBART max context 1024 tokens — long documents are truncated",
            "Qualitative evaluation requires human review (not automated)",
            "BERTScore measured on subset (max 500 samples)",
            "Latency on CPU without optimization — Phase 3 ONNX will improve this",
        ],
    }

    json_path = output_dir / "phase2_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    logger.info("گزارش JSON ذخیره شد: %s", json_path)

    # ── ذخیره خروجی‌های کیفی ─────────────────────────────
    if qualitative_samples:
        qual_path = output_dir / "qualitative_outputs.json"
        with open(qual_path, "w", encoding="utf-8") as f:
            json.dump(qualitative_samples, f, ensure_ascii=False, indent=2)
        logger.info("نمونه‌های کیفی ذخیره شد: %s", qual_path)

    # ── گزارش Markdown ────────────────────────────────────
    _generate_markdown_report(full_report, output_dir)


def _generate_markdown_report(report: dict, output_dir: Path) -> None:
    """گزارش Markdown خوانا برای ناظر پروژه."""

    targets = report.get("project_targets", {})
    rouge = report.get("rouge_evaluation", {})
    bert = report.get("bertscore_evaluation", {})
    latency = report.get("latency_benchmark", {})
    inf_time = rouge.get("inference_time", {})

    md_lines = [
        "# گزارش ارزیابی فاز ۲ — Fine-Tuned AraBART",
        "",
        f"**تاریخ ارزیابی:** {report.get('evaluated_at', 'N/A')}",
        f"**مدل:** {report.get('model_path', 'N/A')}",
        f"**فاز:** {report.get('phase', 'N/A')}",
        "",
        "---",
        "",
        "## ۱. نتایج ROUGE",
        "",
        "| معیار | نتیجه | هدف | وضعیت |",
        "|-------|-------|-----|--------|",
        f"| ROUGE-1 | {rouge.get('rouge1', 'N/A')} | ≥ 0.45 | {targets.get('rouge1', {}).get('status', '?')} |",
        f"| ROUGE-2 | {rouge.get('rouge2', 'N/A')} | ≥ 0.25 | {targets.get('rouge2', {}).get('status', '?')} |",
        f"| ROUGE-L | {rouge.get('rougeL', 'N/A')} | ≥ 0.40 | {targets.get('rougeL', {}).get('status', '?')} |",
        "",
        f"**تعداد نمونه:** {rouge.get('n_samples', 'N/A')} | **شکست:** {rouge.get('n_failed', 0)}",
        "",
        "---",
        "",
        "## ۲. BERTScore",
        "",
        "| معیار | نتیجه | هدف | وضعیت |",
        "|-------|-------|-----|--------|",
        f"| Precision | {bert.get('precision', 'N/A')} | - | - |",
        f"| Recall | {bert.get('recall', 'N/A')} | - | - |",
        f"| F1 | {bert.get('f1', 'N/A')} | ≥ 0.85 | {targets.get('bertscore_f1', {}).get('status', '?')} |",
        "",
        f"**مدل BERTScore:** {bert.get('model', 'N/A')} | **نمونه:** {bert.get('n_samples', 'N/A')}",
        "",
        "---",
        "",
        "## ۳. Latency Baseline (قبل از بهینه‌سازی)",
        "",
        "| طول ورودی (کلمه) | میانگین (ثانیه) | میانه | حداقل | حداکثر | Phase 2 (<60s) |",
        "|------------------|-----------------|-------|-------|--------|----------------|",
    ]

    for length_key, lat_data in sorted(latency.items(), key=lambda x: int(x[0])):
        if isinstance(lat_data, dict) and "mean_seconds" in lat_data:
            phase2_col = lat_data.get("phase2_requirement_60s", "-")
            md_lines.append(
                f"| {lat_data.get('actual_input_words', length_key)} | "
                f"{lat_data.get('mean_seconds', 'N/A')} | "
                f"{lat_data.get('median_seconds', 'N/A')} | "
                f"{lat_data.get('min_seconds', 'N/A')} | "
                f"{lat_data.get('max_seconds', 'N/A')} | "
                f"{phase2_col} |"
            )

    md_lines += [
        "",
        "---",
        "",
        "## ۴. محدودیت‌های شناخته‌شده",
        "",
    ]

    for limitation in report.get("known_limitations", []):
        md_lines.append(f"- {limitation}")

    md_lines += [
        "",
        "---",
        "",
        "## ۵. گام بعدی",
        "",
        "- **فاز ۳:** Export به ONNX و کوانتیزه‌سازی INT8",
        "- هدف فاز ۳: کاهش latency از زیر ۶۰ ثانیه به زیر ۱۵ ثانیه",
        "- ارزیابی کیفی (بررسی انسانی) روی نمونه‌های qualitative_outputs.json",
        "",
    ]

    md_path = output_dir / "phase2_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("گزارش Markdown ذخیره شد: %s", md_path)


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TASK-203: ارزیابی مدل Fine-Tuned AraBART",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="مسیر مدل Fine-Tuned (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="پوشه دیتاست آماده (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "reports" / "training",
        help="پوشه خروجی گزارش",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="device اجرا (پیش‌فرض: auto)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="فقط 100 نمونه برای تست سریع",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="BERTScore را رد کن (گران‌قیمت است)",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()

    # ── تنظیمات ─────────────────────────────────────────────
    cfg_loader = ConfigLoader()

    model_path = args.model_path or str(
        _PROJECT_ROOT / cfg_loader.get(
            "training", "training.output_dir", "models/finetuned/arabart"
        )
    )

    source_name = cfg_loader.get("training", "dataset.source_name", "xlsum_arabic")
    data_dir = args.data_dir or (
        _PROJECT_ROOT / "data" / "training_ready" / source_name
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    gen_config = cfg_loader.get_section("training", "generation")
    if not gen_config:
        gen_config = {
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "repetition_penalty": 1.2,
            "min_new_tokens": 20,
            "max_new_tokens": 128,
        }

    max_source_length = cfg_loader.get("training", "model.max_source_length", 1024)

    eval_cfg = cfg_loader.get_section("training", "evaluation")
    bertscore_model = (eval_cfg or {}).get(
        "bertscore_model", "bert-base-multilingual-cased"
    )
    bertscore_max_samples = (eval_cfg or {}).get("bertscore_max_samples", 500)
    input_lengths = (eval_cfg or {}).get(
        "benchmark_input_lengths", [300, 500, 1000, 1500, 2000]
    )
    warmup_runs = (eval_cfg or {}).get("benchmark_warmup_runs", 2)
    timed_runs = (eval_cfg or {}).get("benchmark_timed_runs", 5)

    max_eval_samples = 100 if args.quick else None

    # ── بارگذاری دیتاست test ────────────────────────────────
    test_file = data_dir / "test.jsonl"
    if not test_file.exists():
        # fallback به processed
        test_file = (
            _PROJECT_ROOT
            / "data"
            / "processed"
            / source_name
            / "test.jsonl"
        )

    if not test_file.exists():
        logger.error(
            "فایل test پیدا نشد: %s\n"
            "ابتدا prepare_dataset.py را اجرا کنید.",
            test_file,
        )
        sys.exit(1)

    test_data = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    logger.info("Test set: %d نمونه از %s", len(test_data), test_file)

    # ── بارگذاری مدل ────────────────────────────────────────
    model, tokenizer = load_model_for_eval(model_path, device)

    # ── ROUGE Evaluation ─────────────────────────────────────
    logger.info("══ ارزیابی ROUGE ══")
    rouge_stats, rouge_results = evaluate_rouge(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=device,
        gen_config=gen_config,
        max_source_length=max_source_length,
        max_samples=max_eval_samples,
    )

    # ── BERTScore ────────────────────────────────────────────
    bertscore_stats = {}
    if not args.skip_bertscore and rouge_results:
        logger.info("══ ارزیابی BERTScore ══")
        predictions = [r["prediction"] for r in rouge_results]
        references = [r["reference"] for r in rouge_results if r["reference"]]

        if len(predictions) == len(references):
            bertscore_stats = evaluate_bertscore(
                predictions=predictions,
                references=references,
                bertscore_model=bertscore_model,
                max_samples=bertscore_max_samples,
            )
        else:
            logger.warning("tعداد predictions و references یکسان نیست.")
    else:
        logger.info("BERTScore رد شد.")

    # ── Latency Benchmark ────────────────────────────────────
    logger.info("══ Latency Benchmark ══")
    latency_stats = run_latency_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        gen_config=gen_config,
        test_data=test_data,
        input_lengths=input_lengths,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        max_source_length=max_source_length,
    )

    # ── Qualitative Evaluation ───────────────────────────────
    logger.info("══ ارزیابی کیفی ══")
    qualitative_samples = qualitative_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=device,
        gen_config=gen_config,
        n_samples=10,
        max_source_length=max_source_length,
    )

    # ── گزارش نهایی ─────────────────────────────────────────
    generate_evaluation_report(
        rouge_stats=rouge_stats,
        bertscore_stats=bertscore_stats,
        latency_stats=latency_stats,
        qualitative_samples=qualitative_samples,
        model_path=model_path,
        output_dir=args.output_dir,
    )

    # ── خلاصه نهایی ──────────────────────────────────────────
    logger.info("══ خلاصه نهایی فاز ۲ ══")
    logger.info(
        "ROUGE-1: %.4f | ROUGE-2: %.4f | ROUGE-L: %.4f",
        rouge_stats.get("rouge1", 0),
        rouge_stats.get("rouge2", 0),
        rouge_stats.get("rougeL", 0),
    )
    if bertscore_stats.get("f1"):
        logger.info("BERTScore F1: %.4f", bertscore_stats["f1"])

    lat_1000 = latency_stats.get("1000", {})
    if lat_1000.get("mean_seconds"):
        logger.info(
            "Latency ~1000 کلمه: %.2f ثانیه | Phase 2 (<60s): %s",
            lat_1000["mean_seconds"],
            lat_1000.get("phase2_requirement_60s", "UNKNOWN"),
        )


if __name__ == "__main__":
    main()