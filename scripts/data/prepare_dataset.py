"""
تبدیل دیتاست‌های خام به فرمت استاندارد برای Fine-tuning.

اصلاح مهم نسبت به نسخه قبل:
    ❌ قبلی: merge همه split ها → shuffle → تقسیم جدید
    ✅ جدید: حفظ split های رسمی dataset

    دلیل: XL-Sum و سایر dataset های استاندارد split های رسمی دارند.
    ترکیب و shuffle مجدد باعث data leakage می‌شود و مقایسه
    با نتایج مقالات را غیرممکن می‌کند.

مراحل برای هر split جداگانه:
1. بارگذاری split رسمی (train/validation/test)
2. فیلترینگ نمونه‌های نامعتبر
3. deduplication بعد از normalization (نه روی متن خام)
4. اجرای pipeline پیش‌پردازش روی text و summary
5. ذخیره split در همان پوشه

اجرا:
    python scripts/data/prepare_dataset.py
    python scripts/data/prepare_dataset.py --source xlsum_arabic
    python scripts/data/prepare_dataset.py --source xlsum_arabic --no-preprocess-summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from arabic_summarizer.preprocessing import ArabicPreprocessingPipeline
from arabic_summarizer.preprocessing.normalizer import ArabicNormalizer
from arabic_summarizer.exceptions import InvalidInputError
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# split های رسمی که باید حفظ شوند
OFFICIAL_SPLITS = ["train", "validation", "test"]


# ── ابزارهای کمکی ────────────────────────────────────────────────

def _count_words(text: str) -> int:
    return len(text.split())


def _normalize_for_hash(text: str, normalizer: ArabicNormalizer) -> str:
    """
    متن را برای محاسبه hash نرمال می‌کند.

    اصلاح مهم: deduplication باید بعد از normalization انجام شود.
    مثال: 'أحمد ذهب' و 'احمد ذهب' بعد از normalize یکسان می‌شوند
    و باید به عنوان تکراری شناخته شوند.
    """
    normalized = normalizer.normalize(text, normalize_teh=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_jsonl(file_path: Path) -> list[dict]:
    """بارگذاری فایل JSONL."""
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("خطا در خط %d: %s", line_num, e)
    return records


def save_jsonl(records: list[dict], file_path: Path) -> None:
    """ذخیره رکوردها در فایل JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── فیلترینگ و deduplication ────────────────────────────────────

def filter_and_deduplicate(
    records: list[dict],
    normalizer: ArabicNormalizer,
    seen_hashes: set[str],
    split_name: str,
    min_words: int = 300,
    max_words: int = 3000,
) -> tuple[list[dict], dict]:
    """
    فیلتر نمونه‌های نامعتبر و حذف تکراری‌ها.

    نکته مهم: seen_hashes بین split ها share می‌شود تا
    تکراری‌های cross-split هم حذف شوند.

    Args:
        records    : رکوردهای ورودی
        normalizer : برای hash بعد از normalize
        seen_hashes: هش‌های دیده‌شده (shared بین split ها)
        split_name : نام split برای لاگ
        min_words  : حداقل کلمات
        max_words  : حداکثر کلمات

    Returns:
        (رکوردهای معتبر، آمار فیلترینگ)
    """
    stats = {
        "split": split_name,
        "total_input": len(records),
        "removed_missing_fields": 0,
        "removed_out_of_range": 0,
        "removed_duplicates": 0,
        "valid": 0,
    }

    valid_records = []

    for record in records:
        text = record.get("text", "").strip()
        summary = record.get("summary", "").strip()

        # بررسی وجود هر دو فیلد
        if not text or not summary:
            stats["removed_missing_fields"] += 1
            continue

        # بررسی محدوده طول
        word_count = _count_words(text)
        if word_count < min_words or word_count > max_words:
            stats["removed_out_of_range"] += 1
            continue

        # deduplication بعد از normalization
        normalized_hash = _normalize_for_hash(text, normalizer)
        if normalized_hash in seen_hashes:
            stats["removed_duplicates"] += 1
            continue

        seen_hashes.add(normalized_hash)
        valid_records.append({
            "text": text,
            "summary": summary,
            "word_count": word_count,
            "text_hash": normalized_hash,
            "source": record.get("source", "unknown"),
            "id": record.get("id", ""),
        })

    stats["valid"] = len(valid_records)
    return valid_records, stats


# ── پیش‌پردازش ──────────────────────────────────────────────────

def preprocess_records(
    records: list[dict],
    pipeline: ArabicPreprocessingPipeline,
    preprocess_summary: bool = True,
    split_name: str = "",
) -> tuple[list[dict], int]:
    """
    اجرای pipeline پیش‌پردازش روی همه رکوردها.

    اصلاح مهم: summary هم normalize می‌شود.
    دلیل: اگر text normalize شود ولی summary خام بماند،
    ROUGE score های آموزش و ارزیابی با هم ناسازگار می‌شوند.

    Args:
        records           : رکوردهای فیلترشده
        pipeline          : pipeline پیش‌پردازش
        preprocess_summary: آیا summary هم normalize شود
        split_name        : نام split برای tqdm

    Returns:
        (رکوردهای پردازش‌شده، تعداد رکوردهای حذف‌شده)
    """
    normalizer = ArabicNormalizer()
    processed = []
    errors = 0

    desc = f"پیش‌پردازش {split_name}" if split_name else "پیش‌پردازش"

    for record in tqdm(records, desc=desc):
        try:
            # پردازش text
            result = pipeline.run(record["text"])
            processed_text = result.cleaned_text
            word_count = result.word_count

            # پردازش summary
            if preprocess_summary:
                # فقط normalization روی summary - بدون validation طول
                processed_summary = normalizer.normalize(
                    record["summary"], normalize_teh=False
                )
            else:
                processed_summary = record["summary"]

            processed.append({
                "text": processed_text,
                "summary": processed_summary,
                "word_count": word_count,
                "text_hash": record["text_hash"],
                "source": record.get("source", "unknown"),
                "id": record.get("id", ""),
            })

        except InvalidInputError:
            # بعد از پیش‌پردازش ممکن است طول تغییر کرده باشد
            errors += 1
        except Exception as exc:
            logger.warning("خطا در پیش‌پردازش رکورد: %s", exc)
            errors += 1

    return processed, errors


# ── پردازش یک سورس ──────────────────────────────────────────────

def process_source(
    source_dir: Path,
    pipeline: ArabicPreprocessingPipeline,
    preprocess_summary: bool = True,
) -> None:
    """
    یک سورس دیتاست را پردازش و ذخیره می‌کند.

    اصل مهم: هر split رسمی جداگانه پردازش می‌شود.
    split های رسمی (train/validation/test) ترکیب نمی‌شوند.

    برای deduplication cross-split، seen_hashes بین همه split ها
    share می‌شود تا یک نمونه در چند split نباشد.
    """
    logger.info("پردازش سورس: %s", source_dir.name)

    # normalizer برای hash
    normalizer = ArabicNormalizer()

    # seen_hashes بین همه split ها share می‌شود
    # این از data leakage cross-split جلوگیری می‌کند
    seen_hashes: set[str] = set()

    out_dir = PROCESSED_DIR / source_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    total_processed = {split: 0 for split in OFFICIAL_SPLITS}

    # ── پردازش هر split به ترتیب (train اول، بعد validation، بعد test) ──
    # دلیل ترتیب: در صورت تکراری، نمونه در train نگه داشته می‌شود
    for split_name in OFFICIAL_SPLITS:
        jsonl_file = source_dir / f"{split_name}.jsonl"

        if not jsonl_file.exists():
            logger.info("  split '%s' وجود ندارد، رد شد.", split_name)
            continue

        logger.info("  پردازش split: %s", split_name)
        records = load_jsonl(jsonl_file)
        logger.info("    بارگذاری %d رکورد", len(records))

        # فیلترینگ + deduplication
        valid_records, filter_stats = filter_and_deduplicate(
            records=records,
            normalizer=normalizer,
            seen_hashes=seen_hashes,
            split_name=split_name,
        )
        logger.info(
            "    فیلترینگ: ورودی=%d | حذف بازه=%d | تکراری=%d | معتبر=%d",
            filter_stats["total_input"],
            filter_stats["removed_out_of_range"],
            filter_stats["removed_duplicates"],
            filter_stats["valid"],
        )

        if not valid_records:
            logger.warning("    هیچ رکورد معتبری در %s باقی نماند.", split_name)
            all_stats.append(filter_stats)
            continue

        # پیش‌پردازش
        processed_records, error_count = preprocess_records(
            records=valid_records,
            pipeline=pipeline,
            preprocess_summary=preprocess_summary,
            split_name=split_name,
        )
        logger.info(
            "    پیش‌پردازش: معتبر=%d | خطا=%d",
            len(processed_records),
            error_count,
        )

        # ذخیره split
        out_file = out_dir / f"{split_name}.jsonl"
        save_jsonl(processed_records, out_file)
        total_processed[split_name] = len(processed_records)

        filter_stats["after_preprocessing"] = len(processed_records)
        filter_stats["preprocessing_errors"] = error_count
        all_stats.append(filter_stats)

    # ── ذخیره آمار ──────────────────────────────────────────────
    stats = {
        "source": source_dir.name,
        "splits": {
            split: {"count": total_processed[split]}
            for split in OFFICIAL_SPLITS
            if total_processed[split] > 0
        },
        "total": sum(total_processed.values()),
        "split_stats": all_stats,
        "methodology": {
            "split_preservation": "official splits preserved (no merge/reshuffle)",
            "deduplication": "post-normalization SHA-256 hash, cross-split shared",
            "summary_preprocessing": preprocess_summary,
        },
    }

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        "دیتاست ذخیره شد در %s | train=%d | val=%d | test=%d",
        out_dir,
        total_processed["train"],
        total_processed["validation"],
        total_processed["test"],
    )


# ── main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="آماده‌سازی دیتاست برای Fine-tuning (با حفظ split های رسمی)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="نام پوشه سورس در data/raw/ یا 'all' برای همه",
    )
    parser.add_argument(
        "--no-preprocess-summary",
        action="store_true",
        default=False,
        help="اگر مشخص شود، summary normalize نمی‌شود",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

    preprocess_summary = not args.no_preprocess_summary

    for source_dir in sorted(source_dirs):
        if source_dir.exists() and source_dir.is_dir():
            process_source(
                source_dir=source_dir,
                pipeline=pipeline,
                preprocess_summary=preprocess_summary,
            )
        else:
            logger.warning("پوشه وجود ندارد: %s", source_dir)

    logger.info("آماده‌سازی دیتاست کامل شد.")


if __name__ == "__main__":
    main()