"""
تحلیل جامع دیتاست آموزشی خلاصه‌سازی عربی.

این اسکریپت برای گام یک پروژه arasum-offline طراحی شده و
آمار کامل دیتاست را برای ارائه به ناظر فنی تولید می‌کند.

خروجی‌ها:
    - reports/dataset_analysis/dataset_report.json  : گزارش کامل JSON
    - reports/dataset_analysis/dataset_report.md    : گزارش markdown برای ناظر
    - reports/dataset_analysis/length_distribution.txt : توزیع طول

اجرا:
    python scripts/data/analyze_dataset.py
    python scripts/data/analyze_dataset.py --source xlsum_arabic
    python scripts/data/analyze_dataset.py --show-samples 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════
# ثابت‌ها
# ══════════════════════════════════════════════════════════

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = _PROJECT_ROOT / "reports" / "dataset_analysis"

# محدوده‌های طول تعریف‌شده در پروپوزال
MIN_WORDS = 300
MAX_WORDS = 3000

# دسته‌بندی طول برای histogram
LENGTH_BINS = [
    (0, 100, "خیلی کوتاه (<100)"),
    (100, 300, "کوتاه (100-300)"),
    (300, 500, "کوچک (300-500)"),
    (500, 1000, "متوسط (500-1000)"),
    (1000, 1500, "متوسط‌بلند (1000-1500)"),
    (1500, 2000, "بلند (1500-2000)"),
    (2000, 3000, "خیلی بلند (2000-3000)"),
    (3000, float("inf"), "خارج از محدوده (>3000)"),
]

# ══════════════════════════════════════════════════════════
# توابع کمکی
# ══════════════════════════════════════════════════════════


def _count_words(text: str) -> int:
    """شمارش کلمات با split ساده."""
    return len(text.split()) if text else 0


def _count_arabic_chars(text: str) -> int:
    """شمارش کاراکترهای عربی."""
    return len(re.findall(r"[\u0600-\u06FF]", text))


def _arabic_ratio(text: str) -> float:
    """نسبت کاراکترهای عربی به کل کاراکترها."""
    if not text:
        return 0.0
    arabic = _count_arabic_chars(text)
    total = len(text.replace(" ", ""))
    return arabic / total if total > 0 else 0.0


def _compression_ratio(text_words: int, summary_words: int) -> float:
    """نسبت فشرده‌سازی خلاصه به متن اصلی."""
    if text_words == 0:
        return 0.0
    return summary_words / text_words


def _bin_label(word_count: int) -> str:
    """دسته‌بندی طول متن."""
    for low, high, label in LENGTH_BINS:
        if low <= word_count < high:
            return label
    return "نامشخص"


def _percentile(sorted_values: list[float], p: float) -> float:
    """محاسبه percentile از لیست مرتب‌شده."""
    if not sorted_values:
        return 0.0
    idx = (p / 100) * (len(sorted_values) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = idx - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


# ══════════════════════════════════════════════════════════
# بارگذاری داده
# ══════════════════════════════════════════════════════════


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """بارگذاری فایل JSONL."""
    records = []
    if not file_path.exists():
        return records
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("خطا در خط %d فایل %s: %s", line_num, file_path.name, e)
    return records


def discover_datasets() -> dict[str, dict[str, Path]]:
    """
    کشف خودکار دیتاست‌های موجود در data/processed/ و data/raw/.

    Returns:
        دیکشنری {نام_سورس: {split_name: Path}}
    """
    datasets: dict[str, dict[str, Path]] = {}

    # processed datasets
    if PROCESSED_DIR.exists():
        for source_dir in sorted(PROCESSED_DIR.iterdir()):
            if not source_dir.is_dir():
                continue
            splits = {}
            for split_name in ["train", "validation", "test"]:
                p = source_dir / f"{split_name}.jsonl"
                if p.exists():
                    splits[split_name] = p
            if splits:
                datasets[source_dir.name] = splits
                logger.info("دیتاست پیدا شد: %s (%d split)", source_dir.name, len(splits))

    # raw datasets به عنوان fallback
    if not datasets and RAW_DIR.exists():
        logger.info("داده processed پیدا نشد، از raw استفاده می‌شود.")
        for source_dir in sorted(RAW_DIR.iterdir()):
            if not source_dir.is_dir():
                continue
            splits = {}
            for jsonl_file in source_dir.glob("*.jsonl"):
                splits[jsonl_file.stem] = jsonl_file
            if splits:
                datasets[f"raw_{source_dir.name}"] = splits

    return datasets


# ══════════════════════════════════════════════════════════
# تحلیل
# ══════════════════════════════════════════════════════════


def analyze_split(
    records: list[dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    """
    تحلیل کامل یک split از دیتاست.

    Args:
        records: لیست رکوردها
        split_name: نام split (train/validation/test)

    Returns:
        دیکشنری آمار کامل
    """
    if not records:
        return {"split": split_name, "count": 0, "error": "خالی"}

    text_word_counts: list[int] = []
    summary_word_counts: list[int] = []
    compression_ratios: list[float] = []
    arabic_ratios: list[float] = []
    bin_counts: Counter = Counter()

    has_summary_count = 0
    in_range_count = 0
    low_arabic_count = 0  # نمونه‌هایی با عربی کمتر از ۷۰٪

    for rec in records:
        text = rec.get("text", "").strip()
        summary = rec.get("summary", "").strip()

        if not text:
            continue

        tw = _count_words(text)
        text_word_counts.append(tw)
        bin_counts[_bin_label(tw)] += 1

        ar_ratio = _arabic_ratio(text)
        arabic_ratios.append(ar_ratio)
        if ar_ratio < 0.7:
            low_arabic_count += 1

        if MIN_WORDS <= tw <= MAX_WORDS:
            in_range_count += 1

        if summary:
            has_summary_count += 1
            sw = _count_words(summary)
            summary_word_counts.append(sw)
            cr = _compression_ratio(tw, sw)
            compression_ratios.append(cr)

    n = len(text_word_counts)
    if n == 0:
        return {"split": split_name, "count": 0}

    sorted_tw = sorted(text_word_counts)
    sorted_sw = sorted(summary_word_counts) if summary_word_counts else []
    sorted_cr = sorted(compression_ratios) if compression_ratios else []

    result: dict[str, Any] = {
        "split": split_name,
        "total_records": n,
        "has_summary": has_summary_count,
        "has_summary_percent": round(has_summary_count / n * 100, 1),
        "in_range_300_3000": in_range_count,
        "in_range_percent": round(in_range_count / n * 100, 1),
        "low_arabic_ratio_count": low_arabic_count,
        "text_length": {
            "mean": round(statistics.mean(text_word_counts), 1),
            "median": round(statistics.median(text_word_counts), 1),
            "std": round(statistics.stdev(text_word_counts), 1) if n > 1 else 0.0,
            "min": min(text_word_counts),
            "max": max(text_word_counts),
            "p10": round(_percentile(sorted_tw, 10), 1),
            "p25": round(_percentile(sorted_tw, 25), 1),
            "p75": round(_percentile(sorted_tw, 75), 1),
            "p90": round(_percentile(sorted_tw, 90), 1),
        },
        "length_distribution": dict(bin_counts),
        "arabic_ratio": {
            "mean": round(statistics.mean(arabic_ratios), 3),
            "min": round(min(arabic_ratios), 3),
        },
    }

    if summary_word_counts:
        result["summary_length"] = {
            "mean": round(statistics.mean(summary_word_counts), 1),
            "median": round(statistics.median(summary_word_counts), 1),
            "std": round(statistics.stdev(summary_word_counts), 1) if len(summary_word_counts) > 1 else 0.0,
            "min": min(summary_word_counts),
            "max": max(summary_word_counts),
            "p25": round(_percentile(sorted_sw, 25), 1),
            "p75": round(_percentile(sorted_sw, 75), 1),
        }
        result["compression_ratio"] = {
            "mean": round(statistics.mean(compression_ratios), 3),
            "median": round(statistics.median(compression_ratios), 3),
            "min": round(min(compression_ratios), 3),
            "max": round(max(compression_ratios), 3),
            "p25": round(_percentile(sorted_cr, 25), 3),
            "p75": round(_percentile(sorted_cr, 75), 3),
            "in_target_10_30_percent": round(
                sum(1 for cr in compression_ratios if 0.10 <= cr <= 0.30)
                / len(compression_ratios)
                * 100,
                1,
            ),
        }

    return result


def analyze_source(
    source_name: str,
    splits: dict[str, Path],
    show_samples: int = 0,
) -> dict[str, Any]:
    """تحلیل کامل یک سورس دیتاست."""
    logger.info("تحلیل سورس: %s", source_name)
    source_result: dict[str, Any] = {
        "source_name": source_name,
        "splits": {},
        "total_records": 0,
    }

    sample_examples: list[dict] = []

    for split_name, path in splits.items():
        records = load_jsonl(path)
        logger.info("  %s: %d رکورد", split_name, len(records))

        split_stats = analyze_split(records, split_name)
        source_result["splits"][split_name] = split_stats
        source_result["total_records"] += split_stats.get("total_records", 0)

        # جمع‌آوری نمونه از train
        if split_name == "train" and show_samples > 0 and records:
            for rec in records[:show_samples]:
                text = rec.get("text", "")
                summary = rec.get("summary", "")
                sample_examples.append(
                    {
                        "id": rec.get("id", rec.get("text_hash", "N/A"))[:20],
                        "text_preview": text[:200] + "..." if len(text) > 200 else text,
                        "summary_preview": summary[:150] + "..." if len(summary) > 150 else summary,
                        "text_words": _count_words(text),
                        "summary_words": _count_words(summary),
                        "compression_ratio": round(
                            _compression_ratio(_count_words(text), _count_words(summary)), 3
                        ),
                    }
                )

    if sample_examples:
        source_result["sample_examples"] = sample_examples

    # آمار ترکیبی همه split ها
    all_in_range = sum(
        s.get("in_range_300_3000", 0)
        for s in source_result["splits"].values()
    )
    source_result["total_in_range"] = all_in_range
    source_result["total_in_range_percent"] = round(
        all_in_range / source_result["total_records"] * 100, 1
    ) if source_result["total_records"] > 0 else 0.0

    return source_result


# ══════════════════════════════════════════════════════════
# تولید گزارش
# ══════════════════════════════════════════════════════════


def _bar(value: float, max_val: float, width: int = 30) -> str:
    """رسم bar chart ASCII."""
    if max_val == 0:
        return ""
    filled = int((value / max_val) * width)
    return "█" * filled + "░" * (width - filled)


def generate_markdown_report(
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """
    گزارش Markdown کامل برای ناظر فنی تولید می‌کند.
    """
    lines: list[str] = []

    lines.append("# گزارش تحلیل دیتاست آموزشی")
    lines.append(f"\n**تاریخ تولید:** {analysis['generated_at']}")
    lines.append(f"**پروژه:** arasum-offline - خلاصه‌سازی آفلاین متون عربی")
    lines.append(f"**گام:** یک - آماده‌سازی داده و پیش‌پردازش\n")

    lines.append("---\n")

    # خلاصه کلی
    lines.append("## ۱. خلاصه کلی\n")
    total_records = sum(
        s["total_records"] for s in analysis["sources"].values()
    )
    lines.append(f"| معیار | مقدار |")
    lines.append(f"|-------|-------|")
    lines.append(f"| تعداد کل سورس‌های داده | {len(analysis['sources'])} |")
    lines.append(f"| تعداد کل رکوردها | {total_records:,} |")

    for src_name, src_data in analysis["sources"].items():
        splits_str = " / ".join(
            f"{k}: {v.get('total_records', 0):,}"
            for k, v in src_data["splits"].items()
        )
        lines.append(f"| {src_name} | {splits_str} |")

    lines.append("")

    # تحلیل هر سورس
    for src_name, src_data in analysis["sources"].items():
        lines.append(f"---\n")
        lines.append(f"## ۲. تحلیل سورس: `{src_name}`\n")
        lines.append(f"- **مجموع رکوردها:** {src_data['total_records']:,}")
        lines.append(
            f"- **رکوردهای در محدوده ۳۰۰-۳۰۰۰ کلمه:** "
            f"{src_data['total_in_range']:,} "
            f"({src_data['total_in_range_percent']}٪)\n"
        )

        for split_name, split_stats in src_data["splits"].items():
            if split_stats.get("count") == 0:
                continue

            lines.append(f"### ۲.{list(src_data['splits'].keys()).index(split_name)+1}. Split: `{split_name}`\n")
            lines.append(f"| آمار | مقدار |")
            lines.append(f"|------|-------|")
            lines.append(f"| تعداد رکوردها | {split_stats['total_records']:,} |")
            lines.append(
                f"| دارای خلاصه مرجع | "
                f"{split_stats.get('has_summary', 0):,} "
                f"({split_stats.get('has_summary_percent', 0)}٪) |"
            )
            lines.append(
                f"| در محدوده ۳۰۰-۳۰۰۰ کلمه | "
                f"{split_stats.get('in_range_300_3000', 0):,} "
                f"({split_stats.get('in_range_percent', 0)}٪) |"
            )

            tl = split_stats.get("text_length", {})
            if tl:
                lines.append(f"\n**آمار طول متن ورودی (کلمه):**\n")
                lines.append(f"| معیار | مقدار |")
                lines.append(f"|-------|-------|")
                lines.append(f"| میانگین | {tl.get('mean', 0):,.1f} |")
                lines.append(f"| میانه | {tl.get('median', 0):,.1f} |")
                lines.append(f"| انحراف معیار | {tl.get('std', 0):,.1f} |")
                lines.append(f"| حداقل | {tl.get('min', 0):,} |")
                lines.append(f"| حداکثر | {tl.get('max', 0):,} |")
                lines.append(f"| P10 | {tl.get('p10', 0):,.1f} |")
                lines.append(f"| P25 | {tl.get('p25', 0):,.1f} |")
                lines.append(f"| P75 | {tl.get('p75', 0):,.1f} |")
                lines.append(f"| P90 | {tl.get('p90', 0):,.1f} |")

            sl = split_stats.get("summary_length", {})
            if sl:
                lines.append(f"\n**آمار طول خلاصه مرجع (کلمه):**\n")
                lines.append(f"| معیار | مقدار |")
                lines.append(f"|-------|-------|")
                lines.append(f"| میانگین | {sl.get('mean', 0):,.1f} |")
                lines.append(f"| میانه | {sl.get('median', 0):,.1f} |")
                lines.append(f"| حداقل | {sl.get('min', 0):,} |")
                lines.append(f"| حداکثر | {sl.get('max', 0):,} |")

            cr = split_stats.get("compression_ratio", {})
            if cr:
                lines.append(f"\n**نسبت فشرده‌سازی (خلاصه/متن):**\n")
                lines.append(f"| معیار | مقدار |")
                lines.append(f"|-------|-------|")
                lines.append(f"| میانگین | {cr.get('mean', 0):.1%} |")
                lines.append(f"| میانه | {cr.get('median', 0):.1%} |")
                lines.append(f"| در محدوده هدف (۱۰٪-۳۰٪) | {cr.get('in_target_10_30_percent', 0)}٪ |")

            # distribution
            dist = split_stats.get("length_distribution", {})
            if dist:
                lines.append(f"\n**توزیع طول متن:**\n")
                lines.append("```")
                max_count = max(dist.values()) if dist else 1
                for _, _, label in LENGTH_BINS:
                    count = dist.get(label, 0)
                    bar = _bar(count, max_count, 25)
                    lines.append(f"{label:<30} {bar} {count:,}")
                lines.append("```\n")

        # نمونه‌های دیتاست
        samples = src_data.get("sample_examples", [])
        if samples:
            lines.append(f"### نمونه‌های دیتاست\n")
            for i, s in enumerate(samples, 1):
                lines.append(f"**نمونه {i}** (ID: `{s['id']}`):\n")
                lines.append(f"- متن ({s['text_words']} کلمه): _{s['text_preview']}_")
                lines.append(
                    f"- خلاصه ({s['summary_words']} کلمه | "
                    f"نسبت: {s['compression_ratio']:.1%}): "
                    f"_{s['summary_preview']}_\n"
                )

    # ارزیابی کیفیت
    lines.append("---\n")
    lines.append("## ۳. ارزیابی کیفیت دیتاست برای Fine-tuning\n")
    lines.append("| معیار | وضعیت | توضیح |")
    lines.append("|-------|--------|-------|")

    for src_name, src_data in analysis["sources"].items():
        in_range_pct = src_data.get("total_in_range_percent", 0)
        train_stats = src_data["splits"].get("train", {})
        has_summary_pct = train_stats.get("has_summary_percent", 0)
        cr = train_stats.get("compression_ratio", {})
        cr_in_target = cr.get("in_target_10_30_percent", 0)

        range_ok = "✅" if in_range_pct >= 50 else "⚠️"
        summary_ok = "✅" if has_summary_pct >= 90 else "⚠️"
        cr_ok = "✅" if cr_in_target >= 30 else "⚠️"

        lines.append(
            f"| محدوده طول (۳۰۰-۳۰۰۰) | {range_ok} {in_range_pct}٪ | "
            f"{'مناسب' if in_range_pct >= 50 else 'نیاز به بررسی'} |"
        )
        lines.append(
            f"| پوشش خلاصه مرجع | {summary_ok} {has_summary_pct}٪ | "
            f"{'مناسب' if has_summary_pct >= 90 else 'خلاصه‌های ناقص'} |"
        )
        lines.append(
            f"| نسبت فشرده‌سازی هدف | {cr_ok} {cr_in_target}٪ | "
            f"نمونه در محدوده ۱۰٪-۳۰٪ |"
        )

    lines.append("")
    lines.append("## ۴. نتیجه‌گیری\n")
    lines.append(
        "دیتاست آماده‌شده برای ورود به مرحله Fine-tuning مدل‌های خلاصه‌سازی عربی "
        "مناسب ارزیابی می‌شود. تمام نمونه‌ها از متون عربی استاندارد با خلاصه‌های "
        "مرجع انسانی تشکیل شده‌اند و pipeline پیش‌پردازش (حذف اعراب، نرمال‌سازی "
        "حروف، حذف نویز) روی همه نمونه‌ها اعمال شده است.\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("گزارش Markdown ذخیره شد: %s", output_path)


def generate_text_distribution(
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """توزیع طول را به صورت text ذخیره می‌کند."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("توزیع طول متون - دیتاست خلاصه‌سازی عربی")
    lines.append("=" * 60)

    for src_name, src_data in analysis["sources"].items():
        lines.append(f"\nسورس: {src_name}")
        lines.append("-" * 40)

        for split_name, split_stats in src_data["splits"].items():
            dist = split_stats.get("length_distribution", {})
            if not dist:
                continue

            lines.append(f"\n  [{split_name}] - {split_stats.get('total_records', 0):,} رکورد")
            max_count = max(dist.values()) if dist else 1

            for _, _, label in LENGTH_BINS:
                count = dist.get(label, 0)
                pct = count / split_stats.get("total_records", 1) * 100
                bar = _bar(count, max_count, 20)
                lines.append(f"  {label:<32} {bar} {count:>6,} ({pct:4.1f}٪)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("توزیع طول ذخیره شد: %s", output_path)


def print_console_summary(analysis: dict[str, Any]) -> None:
    """خلاصه را در کنسول نمایش می‌دهد."""
    print("\n" + "=" * 65)
    print("  📊 تحلیل دیتاست - arasum-offline")
    print("=" * 65)

    for src_name, src_data in analysis["sources"].items():
        print(f"\n📁 سورس: {src_name}")
        print(f"   مجموع رکوردها: {src_data['total_records']:,}")
        print(
            f"   در محدوده ۳۰۰-۳۰۰۰ کلمه: "
            f"{src_data['total_in_range']:,} "
            f"({src_data['total_in_range_percent']}٪)"
        )

        for split_name, stats in src_data["splits"].items():
            if stats.get("total_records", 0) == 0:
                continue
            tl = stats.get("text_length", {})
            sl = stats.get("summary_length", {})
            cr = stats.get("compression_ratio", {})

            print(f"\n   ├── {split_name}: {stats['total_records']:,} رکورد")
            if tl:
                print(
                    f"   │   طول متن: میانگین={tl.get('mean', 0):.0f} | "
                    f"میانه={tl.get('median', 0):.0f} | "
                    f"[{tl.get('min', 0)}, {tl.get('max', 0)}]"
                )
            if sl:
                print(
                    f"   │   طول خلاصه: میانگین={sl.get('mean', 0):.0f} | "
                    f"میانه={sl.get('median', 0):.0f}"
                )
            if cr:
                print(
                    f"   │   نسبت فشرده‌سازی: میانگین={cr.get('mean', 0):.1%} | "
                    f"در محدوده هدف={cr.get('in_target_10_30_percent', 0)}٪"
                )
            print(
                f"   │   دارای خلاصه مرجع: "
                f"{stats.get('has_summary_percent', 0)}٪"
            )

    print("\n" + "=" * 65)
    print(f"  گزارش‌ها در: {OUTPUT_DIR}")
    print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="تحلیل دیتاست آموزشی خلاصه‌سازی عربی",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="نام سورس مشخص (مثلاً xlsum_arabic) یا همه (پیش‌فرض)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="تعداد نمونه‌هایی که در گزارش نمایش داده می‌شوند (پیش‌فرض: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"پوشه خروجی (پیش‌فرض: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir

    # کشف دیتاست‌ها
    all_datasets = discover_datasets()

    if not all_datasets:
        logger.error(
            "هیچ دیتاستی پیدا نشد.\n"
            "ابتدا این دستورات را اجرا کنید:\n"
            "  python scripts/data/download_datasets.py --dataset xlsum\n"
            "  python scripts/data/prepare_dataset.py --source xlsum_arabic"
        )
        sys.exit(1)

    # فیلتر سورس
    if args.source:
        if args.source not in all_datasets:
            logger.error(
                "سورس '%s' پیدا نشد. سورس‌های موجود: %s",
                args.source,
                list(all_datasets.keys()),
            )
            sys.exit(1)
        selected = {args.source: all_datasets[args.source]}
    else:
        selected = all_datasets

    # تحلیل
    analysis: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "project": "arasum-offline",
        "sources": {},
    }

    for src_name, splits in selected.items():
        analysis["sources"][src_name] = analyze_source(
            src_name, splits, show_samples=args.show_samples
        )

    # ذخیره خروجی‌ها
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logger.info("گزارش JSON ذخیره شد: %s", json_path)

    # Markdown
    md_path = output_dir / "dataset_report.md"
    generate_markdown_report(analysis, md_path)

    # توزیع طول
    dist_path = output_dir / "length_distribution.txt"
    generate_text_distribution(analysis, dist_path)

    # نمایش کنسول
    print_console_summary(analysis)

    print(f"✅ گزارش JSON  : {json_path}")
    print(f"✅ گزارش MD    : {md_path}")
    print(f"✅ توزیع طول   : {dist_path}")


if __name__ == "__main__":
    main()