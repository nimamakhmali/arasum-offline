"""
دانلود و بررسی اولیه دیتاست‌های خلاصه‌سازی عربی.

دیتاست‌های هدف:
- XL-Sum بخش عربی (BBC Arabic)
- EASC از HuggingFace
- Auto Arabic Summarization

اجرا:
    python scripts/data/download_datasets.py
    python scripts/data/download_datasets.py --dataset xlsum
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
SAMPLE_SIZE_FOR_INSPECTION = 50


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _count_words(text: str) -> int:
    """تعداد توکن‌های متن را با split ساده می‌شمارد."""
    return len(text.split())


def _inspect_samples(
    samples: list[dict], text_col: str, summary_col: str, dataset_name: str
) -> dict:
    """
    آمار پایه دیتاست را محاسبه می‌کند.
    روی حداکثر SAMPLE_SIZE_FOR_INSPECTION نمونه تصادفی اجرا می‌شود.
    """
    subset = random.sample(samples, min(SAMPLE_SIZE_FOR_INSPECTION, len(samples)))

    text_lengths = [_count_words(s[text_col]) for s in subset]
    summary_lengths = [_count_words(s[summary_col]) for s in subset]

    in_range = sum(1 for l in text_lengths if 300 <= l <= 3000)

    stats = {
        "dataset": dataset_name,
        "inspected_samples": len(subset),
        "text_avg_words": round(sum(text_lengths) / len(text_lengths), 1),
        "text_min_words": min(text_lengths),
        "text_max_words": max(text_lengths),
        "summary_avg_words": round(sum(summary_lengths) / len(summary_lengths), 1),
        "in_range_300_3000": in_range,
        "in_range_percent": round(in_range / len(subset) * 100, 1),
    }
    return stats


def download_xlsum() -> dict:
    """
    دانلود بخش عربی XL-Sum از HuggingFace.
    شامل مقالات BBC Arabic با خلاصه رسمی.
    """
    logger.info("دانلود XL-Sum (عربی) ...")
    dataset = load_dataset("csebuetnlp/xlsum", "arabic", trust_remote_code=True)

    save_dir = RAW_DIR / "xlsum_arabic"
    save_dir.mkdir(exist_ok=True)

    split_info = {}

    for split_name, split_data in dataset.items():
        records = [
            {"text": row["text"], "summary": row["summary"], "id": row["id"]}
            for row in tqdm(split_data, desc=f"  پردازش {split_name}")
        ]

        out_file = save_dir / f"{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        split_info[split_name] = len(records)
        logger.info("  %s: %d نمونه → %s", split_name, len(records), out_file)

    all_samples = [
        {"text": row["text"], "summary": row["summary"]}
        for row in dataset["train"]
    ]
    stats = _inspect_samples(all_samples, "text", "summary", "xlsum_arabic")

    logger.info("آمار XL-Sum: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return {"splits": split_info, "stats": stats}


def download_easc() -> dict:
    """
    دانلود EASC از HuggingFace.
    """
    logger.info("دانلود EASC ...")

    try:
        dataset = load_dataset(
            "abdalrahmanshahrour/auto-arabic-summarization", trust_remote_code=True
        )
    except Exception as e:
        logger.error("خطا در دانلود EASC: %s", e)
        return {"error": str(e)}

    save_dir = RAW_DIR / "easc"
    save_dir.mkdir(exist_ok=True)

    split_info = {}

    for split_name, split_data in dataset.items():
        records = []
        for row in tqdm(split_data, desc=f"  پردازش {split_name}"):
            # نام ستون‌ها ممکن است بین دیتاست‌ها متفاوت باشد
            text = row.get("article") or row.get("text") or row.get("body", "")
            summary = row.get("summary") or row.get("abstract", "")
            if text and summary:
                records.append({"text": text, "summary": summary})

        out_file = save_dir / f"{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        split_info[split_name] = len(records)
        logger.info("  %s: %d نمونه → %s", split_name, len(records), out_file)

    if "train" in dataset:
        all_samples = [
            {
                "text": row.get("article") or row.get("text", ""),
                "summary": row.get("summary") or row.get("abstract", ""),
            }
            for row in dataset["train"]
        ]
        stats = _inspect_samples(all_samples, "text", "summary", "easc")
        logger.info("آمار EASC: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        stats = {}

    return {"splits": split_info, "stats": stats}


def save_download_report(results: dict) -> None:
    """گزارش دانلود را در data/raw ذخیره می‌کند."""
    report_path = RAW_DIR / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("گزارش دانلود ذخیره شد: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="دانلود دیتاست‌های خلاصه‌سازی عربی")
    parser.add_argument(
        "--dataset",
        choices=["xlsum", "easc", "all"],
        default="all",
        help="کدام دیتاست دانلود شود",
    )
    args = parser.parse_args()

    _ensure_dirs()
    results: dict = {}

    if args.dataset in ("xlsum", "all"):
        results["xlsum"] = download_xlsum()

    if args.dataset in ("easc", "all"):
        results["easc"] = download_easc()

    save_download_report(results)
    logger.info("دانلود کامل شد.")


if __name__ == "__main__":
    main()