"""
دانلود و بررسی اولیه دیتاست‌های خلاصه‌سازی عربی.

دیتاست‌های هدف (طبق پروپوزال):
- EASC   : Egyptian Arabic Summarization Corpus
- AraBench: مجموعه benchmark عربی برای ارزیابی
- XL-Sum : بخش عربی BBC Arabic (تکمیلی)

درباره AraBench:
    AraBench یک benchmark برای ارزیابی است نه دیتاست آموزشی مستقیم.
    شامل مجموعه‌ای از متون عربی فصیح و محاوره‌ای با reference های
    انسانی است. از آن برای ارزیابی کیفیت خلاصه‌سازی استفاده می‌کنیم.
    داده‌های آن از HuggingFace و منابع دانشگاهی قابل دسترسی است.

اجرا:
    python scripts/data/download_datasets.py
    python scripts/data/download_datasets.py --dataset easc
    python scripts/data/download_datasets.py --dataset arabench
    python scripts/data/download_datasets.py --dataset xlsum
    python scripts/data/download_datasets.py --dataset all
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


# ═══════════════════════════════════════════════════════════
# ابزارهای کمکی
# ═══════════════════════════════════════════════════════════


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _count_words(text: str) -> int:
    return len(text.split())


def _safe_get(row: dict, *keys: str, default: str = "") -> str:
    """
    چند کلید مختلف را امتحان می‌کند و اولی که وجود دارد را برمی‌گرداند.
    برای دیتاست‌هایی که نام ستون‌ها متفاوت است.
    """
    for key in keys:
        value = row.get(key, "")
        if value and str(value).strip():
            return str(value).strip()
    return default


def _inspect_samples(
    samples: list[dict],
    text_col: str,
    summary_col: str,
    dataset_name: str,
) -> dict:
    """
    آمار پایه دیتاست را از روی نمونه‌های تصادفی محاسبه می‌کند.
    """
    if not samples:
        return {"dataset": dataset_name, "error": "نمونه‌ای وجود ندارد"}

    subset = random.sample(samples, min(SAMPLE_SIZE_FOR_INSPECTION, len(samples)))

    text_lengths = [_count_words(str(s.get(text_col, ""))) for s in subset]
    summary_lengths = [_count_words(str(s.get(summary_col, ""))) for s in subset]

    valid_lengths = [l for l in text_lengths if l > 0]
    in_range = sum(1 for l in text_lengths if 300 <= l <= 3000)

    return {
        "dataset": dataset_name,
        "inspected_samples": len(subset),
        "text_avg_words": round(
            sum(valid_lengths) / len(valid_lengths), 1
        ) if valid_lengths else 0,
        "text_min_words": min(valid_lengths) if valid_lengths else 0,
        "text_max_words": max(valid_lengths) if valid_lengths else 0,
        "summary_avg_words": round(
            sum(summary_lengths) / len(summary_lengths), 1
        ) if summary_lengths else 0,
        "in_range_300_3000_count": in_range,
        "in_range_300_3000_percent": round(
            in_range / len(subset) * 100, 1
        ) if subset else 0,
    }


def _save_records(records: list[dict], out_file: Path) -> None:
    """ذخیره رکوردها در فایل JSONL."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════
# دانلود EASC
# ═══════════════════════════════════════════════════════════


def download_easc() -> dict:
    """
    دانلود EASC (Egyptian Arabic Summarization Corpus).

    این دیتاست شامل مقالات عربی با خلاصه‌های انسانی است.
    چند variant آن روی HuggingFace موجود است و همه را امتحان می‌کنیم.
    """
    logger.info("═══ دانلود EASC ═══")

    # لیست منابع احتمالی EASC روی HuggingFace
    # به ترتیب اولویت امتحان می‌شوند
    easc_sources = [
        {
            "repo": "abdalrahmanshahrour/auto-arabic-summarization",
            "text_keys": ["article", "text", "body", "content"],
            "summary_keys": ["summary", "abstract", "highlights"],
        },
        {
            "repo": "Yiyan/arabic_news_summarization",
            "text_keys": ["article", "text"],
            "summary_keys": ["summary", "abstract"],
        },
    ]

    save_dir = RAW_DIR / "easc"
    save_dir.mkdir(exist_ok=True)
    split_info = {}
    stats = {}

    for source in easc_sources:
        repo = source["repo"]
        logger.info("تلاش برای دانلود از: %s", repo)

        try:
            dataset = load_dataset(repo, trust_remote_code=True)

            for split_name, split_data in dataset.items():
                records = []
                for row in tqdm(split_data, desc=f"  {split_name}"):
                    text = _safe_get(row, *source["text_keys"])
                    summary = _safe_get(row, *source["summary_keys"])
                    if text and summary:
                        records.append({
                            "text": text,
                            "summary": summary,
                            "source": repo,
                        })

                out_file = save_dir / f"{split_name}.jsonl"
                _save_records(records, out_file)
                split_info[split_name] = len(records)
                logger.info("  %s: %d نمونه → %s", split_name, len(records), out_file.name)

            # آمار روی train اگر وجود داشت
            if "train" in dataset:
                sample_records = [
                    {
                        "text": _safe_get(row, *source["text_keys"]),
                        "summary": _safe_get(row, *source["summary_keys"]),
                    }
                    for row in dataset["train"]
                ]
                stats = _inspect_samples(sample_records, "text", "summary", "easc")

            logger.info("EASC با موفقیت دانلود شد از: %s", repo)
            break  # موفق بود، ادامه نده

        except Exception as exc:
            logger.warning("ناموفق از %s: %s", repo, exc)
            continue

    if not split_info:
        logger.error(
            "هیچ منبعی برای EASC کار نکرد. "
            "لطفاً دیتاست را دستی از "
            "https://huggingface.co/datasets دانلود کنید."
        )
        return {"error": "دانلود EASC ناموفق بود", "splits": {}, "stats": {}}

    return {"splits": split_info, "stats": stats}


# ═══════════════════════════════════════════════════════════
# دانلود AraBench
# ═══════════════════════════════════════════════════════════


def download_arabench() -> dict:
    """
    دانلود AraBench برای ارزیابی.

    AraBench یک benchmark چندوجهی برای ارزیابی سیستم‌های پردازش
    متن عربی است. بخش summarization آن شامل متون عربی فصیح و
    خلاصه‌های مرجع است که برای محاسبه ROUGE استفاده می‌شود.

    این دیتاست به عنوان test set ارزیابی استفاده می‌شود،
    نه برای آموزش مدل.

    منبع اصلی: https://github.com/CAMeL-Lab/AraBench
    """
    logger.info("═══ دانلود AraBench ═══")

    save_dir = RAW_DIR / "arabench"
    save_dir.mkdir(exist_ok=True)

    split_info = {}
    stats = {}

    # ── تلاش ۱: دانلود از HuggingFace ──────────────────────────
    arabench_hf_sources = [
        {
            "repo": "CAMeL-Lab/arabic_pos_dialect",
            "config": None,
            "text_keys": ["text", "sentence"],
            "summary_keys": ["summary", "label"],
            "note": "بخش MSA (عربی فصیح) AraBench",
        },
    ]

    hf_success = False
    for source in arabench_hf_sources:
        try:
            logger.info("تلاش دانلود AraBench از HuggingFace: %s", source["repo"])
            kwargs = {"trust_remote_code": True}
            if source.get("config"):
                kwargs["name"] = source["config"]

            dataset = load_dataset(source["repo"], **kwargs)

            for split_name, split_data in dataset.items():
                records = []
                for row in tqdm(split_data, desc=f"  {split_name}"):
                    text = _safe_get(row, *source["text_keys"])
                    summary = _safe_get(row, *source["summary_keys"])
                    if text and summary:
                        records.append({
                            "text": text,
                            "summary": summary,
                            "source": source["repo"],
                        })

                if records:
                    out_file = save_dir / f"{split_name}.jsonl"
                    _save_records(records, out_file)
                    split_info[split_name] = len(records)
                    logger.info(
                        "  %s: %d نمونه → %s",
                        split_name, len(records), out_file.name,
                    )

            hf_success = bool(split_info)
            if hf_success:
                break

        except Exception as exc:
            logger.warning("ناموفق: %s", exc)
            continue

    # ── تلاش ۲: استفاده از XL-Sum به عنوان جایگزین AraBench ────
    # XL-Sum بخش عربی آن از BBC Arabic است و کیفیت بسیار بالایی
    # دارد. برای ارزیابی benchmark مناسب است.
    if not hf_success:
        logger.info(
            "AraBench مستقیم در دسترس نیست. "
            "از XL-Sum عربی به عنوان benchmark ارزیابی استفاده می‌شود."
        )
        logger.info(
            "توضیح: AraBench اصلی نیاز به دسترسی مستقیم از "
            "https://github.com/CAMeL-Lab/AraBench دارد. "
            "لطفاً طبق راهنمای docs/architecture.md آن را دستی دانلود کنید."
        )

        try:
            logger.info("دانلود XL-Sum عربی برای benchmark ارزیابی ...")
            dataset = load_dataset(
                "csebuetnlp/xlsum", "arabic", trust_remote_code=True
            )

            # فقط test split را برای benchmark می‌خواهیم
            if "test" in dataset:
                records = [
                    {
                        "text": row["text"],
                        "summary": row["summary"],
                        "id": row.get("id", ""),
                        "source": "xlsum_arabic_benchmark",
                    }
                    for row in tqdm(dataset["test"], desc="  benchmark test")
                ]

                out_file = save_dir / "benchmark_test.jsonl"
                _save_records(records, out_file)
                split_info["benchmark_test"] = len(records)
                logger.info(
                    "  benchmark_test: %d نمونه → %s",
                    len(records), out_file.name,
                )

                stats = _inspect_samples(
                    [{"text": r["text"], "summary": r["summary"]} for r in records],
                    "text",
                    "summary",
                    "arabench_xlsum_fallback",
                )

        except Exception as exc:
            logger.error("خطا در دانلود XL-Sum برای benchmark: %s", exc)
            return {
                "error": str(exc),
                "splits": split_info,
                "stats": stats,
                "note": "AraBench دستی دانلود شود",
            }

    # ── ذخیره راهنمای دانلود دستی ──────────────────────────────
    manual_guide = {
        "title": "راهنمای دانلود دستی AraBench",
        "url": "https://github.com/CAMeL-Lab/AraBench",
        "steps": [
            "1. به آدرس https://github.com/CAMeL-Lab/AraBench بروید",
            "2. فایل‌های dataset را دانلود کنید",
            "3. فایل‌ها را در data/raw/arabench/ قرار دهید",
            "4. اسکریپت prepare_dataset.py را اجرا کنید",
        ],
        "alternative": "XL-Sum عربی به عنوان جایگزین معتبر استفاده شده است",
    }

    guide_path = save_dir / "MANUAL_DOWNLOAD_GUIDE.json"
    with open(guide_path, "w", encoding="utf-8") as f:
        json.dump(manual_guide, f, ensure_ascii=False, indent=2)

    logger.info("راهنمای دانلود دستی ذخیره شد: %s", guide_path)

    if not stats and split_info:
        first_file = save_dir / f"{list(split_info.keys())[0]}.jsonl"
        if first_file.exists():
            sample_records = []
            with open(first_file, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= SAMPLE_SIZE_FOR_INSPECTION:
                        break
                    sample_records.append(json.loads(line))
            stats = _inspect_samples(
                sample_records, "text", "summary", "arabench"
            )

    return {"splits": split_info, "stats": stats}


# ═══════════════════════════════════════════════════════════
# دانلود XL-Sum (تکمیلی)
# ═══════════════════════════════════════════════════════════


def download_xlsum() -> dict:
    """
    دانلود بخش عربی XL-Sum از HuggingFace.
    شامل مقالات BBC Arabic با خلاصه رسمی.
    این دیتاست به عنوان منبع اصلی آموزش استفاده می‌شود.
    """
    logger.info("═══ دانلود XL-Sum (عربی) ═══")

    dataset = load_dataset("csebuetnlp/xlsum", "arabic", trust_remote_code=True)
    save_dir = RAW_DIR / "xlsum_arabic"
    save_dir.mkdir(exist_ok=True)

    split_info = {}
    all_train_samples = []

    for split_name, split_data in dataset.items():
        records = [
            {
                "text": row["text"],
                "summary": row["summary"],
                "id": row.get("id", ""),
                "source": "xlsum_arabic",
            }
            for row in tqdm(split_data, desc=f"  {split_name}")
        ]

        out_file = save_dir / f"{split_name}.jsonl"
        _save_records(records, out_file)
        split_info[split_name] = len(records)
        logger.info("  %s: %d نمونه → %s", split_name, len(records), out_file.name)

        if split_name == "train":
            all_train_samples = [
                {"text": r["text"], "summary": r["summary"]} for r in records
            ]

    stats = _inspect_samples(all_train_samples, "text", "summary", "xlsum_arabic")
    logger.info(
        "آمار XL-Sum: %s",
        json.dumps(stats, ensure_ascii=False, indent=2),
    )

    return {"splits": split_info, "stats": stats}


# ═══════════════════════════════════════════════════════════
# ذخیره گزارش
# ═══════════════════════════════════════════════════════════


def save_download_report(results: dict) -> None:
    """گزارش کامل دانلود را در data/raw ذخیره می‌کند."""
    report_path = RAW_DIR / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("گزارش دانلود ذخیره شد: %s", report_path)


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="دانلود دیتاست‌های خلاصه‌سازی عربی",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["easc", "arabench", "xlsum", "all"],
        default="all",
        help=(
            "کدام دیتاست دانلود شود:\n"
            "  easc     : Egyptian Arabic Summarization Corpus\n"
            "  arabench : AraBench benchmark dataset\n"
            "  xlsum    : XL-Sum بخش عربی (BBC Arabic)\n"
            "  all      : همه دیتاست‌ها (پیش‌فرض)"
        ),
    )
    args = parser.parse_args()

    _ensure_dirs()
    results: dict = {}

    if args.dataset in ("easc", "all"):
        results["easc"] = download_easc()

    if args.dataset in ("arabench", "all"):
        results["arabench"] = download_arabench()

    if args.dataset in ("xlsum", "all"):
        results["xlsum"] = download_xlsum()

    save_download_report(results)

    # نمایش خلاصه نهایی
    logger.info("═══ خلاصه دانلود ═══")
    for dataset_name, result in results.items():
        if "error" in result:
            logger.error("  %s: ❌ %s", dataset_name, result["error"])
        else:
            total = sum(result.get("splits", {}).values())
            logger.info("  %s: ✅ %d نمونه کل", dataset_name, total)

    logger.info("دانلود کامل شد.")


if __name__ == "__main__":
    main()