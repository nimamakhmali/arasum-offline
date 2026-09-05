"""
TASK-201 / TASK-202: آماده‌سازی دیتاست برای Fine-Tuning AraBART.

این اسکریپت دیتاست processed را می‌خواند و به فرمت
HuggingFace Dataset تبدیل می‌کند که Trainer مستقیم استفاده کند.

نکته مهم: این اسکریپت با scripts/data/prepare_dataset.py
متفاوت است. آن اسکریپت داده خام را پاک‌سازی می‌کند.
این اسکریپت داده پاک‌شده را برای tokenization آماده می‌کند.

اجرا:
    python scripts/training/prepare_dataset.py
    python scripts/training/prepare_dataset.py --source xlsum_arabic
    python scripts/training/prepare_dataset.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from arabic_summarizer.utils.config_loader import ConfigLoader
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


def load_jsonl(file_path: Path, max_samples: Optional[int] = None) -> list[dict]:
    """
    فایل JSONL را می‌خواند.

    Args:
        file_path: مسیر فایل
        max_samples: حداکثر تعداد (برای smoke test)

    Returns:
        لیست رکوردها
    """
    records = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("خطا در خواندن خط %d: %s", i + 1, e)
    return records


def verify_record_fields(
    records: list[dict],
    text_field: str,
    summary_field: str,
    split_name: str,
) -> list[dict]:
    """
    رکوردها را از نظر وجود فیلدهای ضروری بررسی می‌کند.

    Args:
        records: رکوردهای ورودی
        text_field: نام فیلد متن
        summary_field: نام فیلد خلاصه
        split_name: نام split برای لاگ

    Returns:
        رکوردهای معتبر
    """
    valid = []
    n_missing_text = 0
    n_missing_summary = 0
    n_empty_text = 0
    n_empty_summary = 0

    for rec in records:
        text = rec.get(text_field, "")
        summary = rec.get(summary_field, "")

        if text_field not in rec:
            n_missing_text += 1
            continue
        if summary_field not in rec:
            n_missing_summary += 1
            continue
        if not text or not text.strip():
            n_empty_text += 1
            continue
        if not summary or not summary.strip():
            n_empty_summary += 1
            continue

        valid.append(rec)

    if n_missing_text or n_missing_summary or n_empty_text or n_empty_summary:
        logger.warning(
            "[%s] فیلد text نداشت: %d | فیلد summary نداشت: %d | "
            "text خالی: %d | summary خالی: %d",
            split_name,
            n_missing_text,
            n_missing_summary,
            n_empty_text,
            n_empty_summary,
        )

    logger.info(
        "[%s] %d رکورد معتبر از %d",
        split_name,
        len(valid),
        len(records),
    )
    return valid


def analyze_length_distribution(
    records: list[dict],
    text_field: str,
    summary_field: str,
    split_name: str,
) -> dict:
    """
    توزیع طول متون را تحلیل و گزارش می‌دهد.

    این تحلیل برای مستند کردن:
    - truncation در tokenization
    - نسبت فشرده‌سازی
    - سازگاری با هدف 10-30%
    """
    text_lengths = [len(r[text_field].split()) for r in records]
    summary_lengths = [len(r[summary_field].split()) for r in records]
    ratios = [
        s / t if t > 0 else 0
        for t, s in zip(text_lengths, summary_lengths)
    ]

    # چند درصد در محدوده 10-30%؟
    in_target = sum(1 for r in ratios if 0.10 <= r <= 0.30)

    stats = {
        "split": split_name,
        "n_records": len(records),
        "text_words": {
            "mean": round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0,
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
        },
        "summary_words": {
            "mean": round(sum(summary_lengths) / len(summary_lengths), 1) if summary_lengths else 0,
            "min": min(summary_lengths) if summary_lengths else 0,
            "max": max(summary_lengths) if summary_lengths else 0,
        },
        "compression_ratio": {
            "mean": round(sum(ratios) / len(ratios), 4) if ratios else 0,
            "in_target_10_30_pct": round(in_target / len(ratios) * 100, 1) if ratios else 0,
        },
    }

    logger.info(
        "[%s] متن: mean=%.0f | خلاصه: mean=%.0f | "
        "نسبت فشرده‌سازی: mean=%.3f | در محدوده 10-30%%: %.1f%%",
        split_name,
        stats["text_words"]["mean"],
        stats["summary_words"]["mean"],
        stats["compression_ratio"]["mean"],
        stats["compression_ratio"]["in_target_10_30_pct"],
    )

    return stats


def prepare_split(
    processed_dir: Path,
    source_name: str,
    split_name: str,
    text_field: str,
    summary_field: str,
    max_samples: Optional[int] = None,
) -> Optional[list[dict]]:
    """
    یک split را می‌خواند و آماده می‌کند.

    Returns:
        لیست رکوردهای آماده یا None اگر فایل وجود نداشت
    """
    file_path = processed_dir / source_name / f"{split_name}.jsonl"

    if not file_path.exists():
        logger.warning("فایل split پیدا نشد: %s", file_path)
        return None

    logger.info("بارگذاری split '%s' از: %s", split_name, file_path)
    records = load_jsonl(file_path, max_samples=max_samples)

    if not records:
        logger.error("فایل خالی است: %s", file_path)
        return None

    # بررسی فیلدها
    valid_records = verify_record_fields(records, text_field, summary_field, split_name)

    if not valid_records:
        logger.error("هیچ رکورد معتبری در %s پیدا نشد.", split_name)
        return None

    # تحلیل توزیع طول
    analyze_length_distribution(valid_records, text_field, summary_field, split_name)

    return valid_records


def save_prepared_split(
    records: list[dict],
    output_dir: Path,
    split_name: str,
    text_field: str,
    summary_field: str,
) -> Path:
    """
    split آماده‌شده را در فرمت استاندارد ذخیره می‌کند.

    فرمت خروجی: JSONL با فیلدهای text و summary
    این فرمت مستقیماً توسط finetune.py خوانده می‌شود.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{split_name}.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for rec in records:
            out_rec = {
                "text": rec[text_field].strip(),
                "summary": rec[summary_field].strip(),
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    logger.info("ذخیره شد: %s (%d رکورد)", out_file, len(records))
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="آماده‌سازی دیتاست برای Fine-Tuning AraBART",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="نام سورس دیتاست (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="فقط 100 نمونه برای تست سریع",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="پوشه خروجی (پیش‌فرض: data/training_ready)",
    )
    args = parser.parse_args()

    cfg = ConfigLoader()
    processed_dir = _PROJECT_ROOT / cfg.get(
        "training", "dataset.processed_dir", "data/processed"
    )
    source_name = args.source or cfg.get(
        "training", "dataset.source_name", "xlsum_arabic"
    )
    text_field = cfg.get("training", "dataset.text_field", "text")
    summary_field = cfg.get("training", "dataset.summary_field", "summary")

    output_dir = args.output_dir or _PROJECT_ROOT / "data" / "training_ready" / source_name

    max_train = 200 if args.smoke_test else cfg.get(
        "training", "dataset.max_train_samples", None
    )
    max_eval = 50 if args.smoke_test else cfg.get(
        "training", "dataset.max_eval_samples", None
    )
    max_test = 50 if args.smoke_test else cfg.get(
        "training", "dataset.max_test_samples", None
    )

    if args.smoke_test:
        logger.info("حالت Smoke Test: max_train=%d, max_eval=%d", max_train, max_eval)

    splits_config = {
        "train": max_train,
        "validation": max_eval,
        "test": max_test,
    }

    all_stats = {}

    for split_name, max_samples in splits_config.items():
        records = prepare_split(
            processed_dir=processed_dir,
            source_name=source_name,
            split_name=split_name,
            text_field=text_field,
            summary_field=summary_field,
            max_samples=max_samples,
        )

        if records is not None:
            save_prepared_split(
                records=records,
                output_dir=output_dir,
                split_name=split_name,
                text_field=text_field,
                summary_field=summary_field,
            )
            all_stats[split_name] = len(records)

    # ذخیره metadata
    import json as json_mod
    from datetime import datetime, timezone

    metadata = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "smoke_test": args.smoke_test,
        "splits": all_stats,
        "fields": {"text": text_field, "summary": summary_field},
        "compression_constraint_note": (
            "XL-Sum compression ratio ≈ 5%, product target 10-30%. "
            "Training on original references. "
            "Output length enforced at inference via generation constraints."
        ),
    }

    meta_file = output_dir / "preparation_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json_mod.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("آماده‌سازی کامل شد. اطلاعات: %s", all_stats)
    logger.info("خروجی در: %s", output_dir)


if __name__ == "__main__":
    main()