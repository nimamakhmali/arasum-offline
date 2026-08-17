"""
تبدیل دیتاست‌های خام به فرمت استاندارد برای Fine-tuning.

مراحل:
1. بارگذاری فایل‌های JSONL از data/raw/
2. فیلترینگ نمونه‌های نامعتبر
3. حذف تکراری با hash
4. اجرای pipeline پیش‌پردازش
5. تقسیم train/validation/test
6. ذخیره در data/processed/

اجرا:
    python scripts/data/prepare_dataset.py
    python scripts/data/prepare_dataset.py --source xlsum_arabic --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tqdm import tqdm

from arabic_summarizer.preprocessing import ArabicPreprocessingPipeline
from arabic_summarizer.exceptions import InvalidInputError
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# نسبت تقسیم دیتاست
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def _hash_text(text: str) -> str:
    """Hash SHA-256 متن برای تشخیص تکراری‌ها."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_words(text: str) -> int:
    return len(text.split())


def load_jsonl(file_path: Path) -> list[dict]:
    """بارگذاری فایل JSONL."""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], file_path: Path) -> None:
    """ذخیره رکوردها در فایل JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_and_deduplicate(
    records: list[dict],
    text_col: str = "text",
    summary_col: str = "summary",
    min_words: int = 300,
    max_words: int = 3000,
) -> tuple[list[dict], dict]:
    """
    فیلتر نمونه‌های نامعتبر و حذف تکراری‌ها.

    Returns:
        (لیست رکوردهای معتبر، آمار فیلترینگ)
    """
    stats = {
        "total_input": len(records),
        "removed_missing_fields": 0,
        "removed_out_of_range": 0,
        "removed_duplicates": 0,
    }

    seen_hashes: set[str] = set()
    valid_records = []

    for record in records:
        text = record.get(text_col, "").strip()
        summary = record.get(summary_col, "").strip()

        # بررسی وجود هر دو فیلد
        if not text or not summary:
            stats["removed_missing_fields"] += 1
            continue

        # بررسی محدوده طول
        word_count = _count_words(text)
        if word_count < min_words or word_count > max_words:
            stats["removed_out_of_range"] += 1
            continue

        # بررسی تکراری
        text_hash = _hash_text(text)
        if text_hash in seen_hashes:
            stats["removed_duplicates"] += 1
            continue

        seen_hashes.add(text_hash)
        valid_records.append(
            {
                "text": text,
                "summary": summary,
                "word_count": word_count,
                "text_hash": text_hash,
            }
        )

    stats["valid_after_filter"] = len(valid_records)
    return valid_records, stats


def preprocess_records(
    records: list[dict], pipeline: ArabicPreprocessingPipeline
) -> tuple[list[dict], int]:
    """
    اجرای pipeline پیش‌پردازش روی همه رکوردها.

    Returns:
        (رکوردهای پردازش‌شده، تعداد رکوردهای حذف‌شده به دلیل خطا)
    """
    processed = []
    errors = 0

    for record in tqdm(records, desc="پیش‌پردازش"):
        try:
            result = pipeline.run(record["text"])
            processed.append(
                {
                    "text": result.cleaned_text,
                    "summary": record["summary"],
                    "word_count": result.word_count,
                    "text_hash": record["text_hash"],
                }
            )
        except InvalidInputError:
            # بعد از پیش‌پردازش ممکن است طول تغییر کرده باشد
            errors += 1
        except Exception as exc:
            logger.warning("خطا در پیش‌پردازش یک رکورد: %s", exc)
            errors += 1

    return processed, errors


def split_dataset(
    records: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    تقسیم دیتاست به train/validation/test.
    از random.shuffle با seed ثابت استفاده می‌کند.
    """
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def compute_dataset_stats(
    train: list[dict], val: list[dict], test: list[dict]
) -> dict:
    """محاسبه آمار نهایی دیتاست برای گزارش."""

    def avg_words(records: list[dict]) -> float:
        if not records:
            return 0.0
        return round(sum(r["word_count"] for r in records) / len(records), 1)

    return {
        "train": {"count": len(train), "avg_words": avg_words(train)},
        "validation": {"count": len(val), "avg_words": avg_words(val)},
        "test": {"count": len(test), "avg_words": avg_words(test)},
        "total": len(train) + len(val) + len(test),
    }


def process_source(source_dir: Path, seed: int, pipeline: ArabicPreprocessingPipeline) -> None:
    """
    یک سورس دیتاست را پردازش و ذخیره می‌کند.
    """
    logger.info("پردازش سورس: %s", source_dir.name)

    # بارگذاری همه split های موجود
    all_records = []
    for jsonl_file in source_dir.glob("*.jsonl"):
        records = load_jsonl(jsonl_file)
        all_records.extend(records)
        logger.info("  بارگذاری %d رکورد از %s", len(records), jsonl_file.name)

    if not all_records:
        logger.warning("هیچ رکوردی در %s پیدا نشد.", source_dir)
        return

    # فیلترینگ و deduplication
    valid_records, filter_stats = filter_and_deduplicate(all_records)
    logger.info("آمار فیلترینگ: %s", json.dumps(filter_stats, ensure_ascii=False))

    # پیش‌پردازش
    processed_records, error_count = preprocess_records(valid_records, pipeline)
    logger.info(
        "پیش‌پردازش کامل شد. معتبر: %d، خطا: %d",
        len(processed_records),
        error_count,
    )

    # تقسیم‌بندی
    train, val, test = split_dataset(processed_records, seed=seed)

    # ذخیره
    out_dir = PROCESSED_DIR / source_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val, out_dir / "validation.jsonl")
    save_jsonl(test, out_dir / "test.jsonl")

    # آمار نهایی
    stats = compute_dataset_stats(train, val, test)
    stats["filter_stats"] = filter_stats
    stats["preprocessing_errors"] = error_count

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        "دیتاست ذخیره شد در %s | train: %d | val: %d | test: %d",
        out_dir,
        len(train),
        len(val),
        len(test),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="آماده‌سازی دیتاست برای Fine-tuning")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="نام پوشه سورس در data/raw/ یا 'all' برای همه",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="عدد seed برای تقسیم‌بندی تصادفی",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # pipeline بدون validation سخت‌گیرانه چون قبلاً فیلتر شده
    pipeline = ArabicPreprocessingPipeline(
        min_words=300,
        max_words=3000,
        normalize_teh=False,
        use_camel_tools=True,
    )

    if args.source == "all":
        source_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    else:
        source_dirs = [RAW_DIR / args.source]

    if not source_dirs:
        logger.error("هیچ سورسی در %s پیدا نشد.", RAW_DIR)
        return

    for source_dir in source_dirs:
        if source_dir.exists():
            process_source(source_dir, args.seed, pipeline)
        else:
            logger.warning("پوشه وجود ندارد: %s", source_dir)

    logger.info("آماده‌سازی دیتاست کامل شد.")


if __name__ == "__main__":
    main()