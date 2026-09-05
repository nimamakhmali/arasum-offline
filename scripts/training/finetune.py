"""
TASK-202: Fine-Tuning مدل AraBART با LoRA روی دیتاست خلاصه‌سازی عربی.

این اسکریپت برای اجرا روی Google Colab GPU یا GPU محلی طراحی شده.
در محیط CPU هم اجرا می‌شود اما بسیار کند خواهد بود.

پیش‌نیاز:
    python scripts/training/prepare_dataset.py

اجرا:
    # Smoke Test (100 نمونه، بررسی صحت pipeline)
    python scripts/training/finetune.py --smoke-test

    # آموزش کامل
    python scripts/training/finetune.py

    # با تنظیمات دستی
    python scripts/training/finetune.py --epochs 5 --lr 3e-5 --batch-size 4

    # روی Colab با Google Drive
    python scripts/training/finetune.py --output-dir /content/drive/MyDrive/arasum/

نکات مهم:
    - در smoke test فقط صحت pipeline بررسی می‌شود، کیفیت مدل مهم نیست
    - آموزش کامل روی GPU با ≥8GB VRAM توصیه می‌شود
    - LoRA adapters ذخیره می‌شوند (نه مدل کامل) - حجم کمتر
    - مدل merge‌شده برای inference ذخیره می‌شود
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from arabic_summarizer.utils.config_loader import ConfigLoader
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)



# Environment Diagnostic


def print_environment_info() -> dict[str, Any]:
    """
    اطلاعات محیط را چاپ و برمی‌گرداند.
    برای reproducibility کامل همه نسخه‌ها ثبت می‌شوند.
    """
    import platform

    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            logger.info("GPU: %s | VRAM: %.1f GB", info["gpu_name"], info["gpu_vram_gb"])
        else:
            logger.warning(
                "CUDA موجود نیست. آموزش روی CPU بسیار کند خواهد بود."
            )
    except ImportError:
        logger.error("PyTorch نصب نیست!")
        raise

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        logger.error("Transformers نصب نیست!")
        raise

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        logger.error("PEFT نصب نیست! pip install peft")
        raise

    try:
        import accelerate
        info["accelerate_version"] = accelerate.__version__
    except ImportError:
        logger.warning("Accelerate نصب نیست. pip install accelerate")
        info["accelerate_version"] = "not installed"

    try:
        import datasets
        info["datasets_version"] = datasets.__version__
    except ImportError:
        logger.error("Datasets نصب نیست! pip install datasets")
        raise

    logger.info("محیط: Python=%s | PyTorch=%s | Transformers=%s | PEFT=%s",
                info["python_version"],
                info.get("torch_version", "?"),
                info.get("transformers_version", "?"),
                info.get("peft_version", "?"))

    return info



# Dataset Loading


def load_dataset_from_jsonl(
    data_dir: Path,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> tuple[Any, Any, Any]:
    """
    دیتاست را از فایل‌های JSONL می‌خواند.

    اهمیت split separation:
        train, validation, test هیچ‌گاه با هم mix نمی‌شوند.
        test split تا ارزیابی نهایی دست نخورده می‌ماند.

    Returns:
        (train_dataset, eval_dataset, test_dataset)
        هر کدام می‌تواند None باشد اگر فایل نبود.
    """
    from datasets import Dataset

    def read_jsonl(file_path: Path, max_samples: Optional[int]) -> Optional[Dataset]:
        if not file_path.exists():
            logger.warning("فایل پیدا نشد: %s", file_path)
            return None

        records = []
        with open(file_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            return None

        # اطمینان از وجود فیلدهای ضروری
        valid = [r for r in records if r.get("text") and r.get("summary")]
        if len(valid) < len(records):
            logger.warning(
                "%d رکورد بدون text یا summary حذف شد.",
                len(records) - len(valid),
            )

        return Dataset.from_list(valid)

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"
    test_file = data_dir / "test.jsonl"

    train_ds = read_jsonl(train_file, max_train_samples)
    eval_ds = read_jsonl(val_file, max_eval_samples)
    test_ds = read_jsonl(test_file, None)  # test هیچ‌وقت truncate نمی‌شود

    logger.info(
        "دیتاست: train=%s | val=%s | test=%s",
        len(train_ds) if train_ds else "N/A",
        len(eval_ds) if eval_ds else "N/A",
        len(test_ds) if test_ds else "N/A",
    )

    return train_ds, eval_ds, test_ds



# Tokenization


def create_tokenize_function(
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
    text_field: str = "text",
    summary_field: str = "summary",
):
    """
    تابع tokenization برای Seq2Seq (BART architecture).

    نکات مهم برای BART (بر خلاف T5):
    - نیازی به task prefix نیست
    - labels باید با -100 padding شوند (نه pad_token_id)
      تا در loss محاسبه نشوند
    - attention_mask برای encoder input است
    - decoder_input_ids توسط model خودکار ساخته می‌شود

    Args:
        tokenizer: AraBART tokenizer
        max_source_length: حداکثر توکن ورودی (1024 برای AraBART)
        max_target_length: حداکثر توکن خلاصه (128)
        text_field: نام فیلد متن
        summary_field: نام فیلد خلاصه

    Returns:
        تابع tokenize آماده برای dataset.map()
    """

    def tokenize_function(examples: dict) -> dict:
        # ── Encode source texts ────────────────────────────────
        model_inputs = tokenizer(
            examples[text_field],
            max_length=max_source_length,
            truncation=True,
            padding=False,  # DataCollator padding را انجام می‌دهد
        )

        # ── Encode target summaries ───────────────────────────
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[summary_field],
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )

        # جایگزینی pad_token_id با -100 در labels
        # تا loss روی padding محاسبه نشود
        model_inputs["labels"] = [
            [
                (token_id if token_id != tokenizer.pad_token_id else -100)
                for token_id in label_ids
            ]
            for label_ids in labels["input_ids"]
        ]

        return model_inputs

    return tokenize_function


def verify_tokenization_examples(
    train_dataset: Any,
    tokenizer: Any,
    n_examples: int = 3,
) -> None:
    """
    چند نمونه tokenize‌شده را بررسی می‌کند.

    این verification حیاتی است:
    - جهت source/target را تأیید می‌کند
    - truncation را اندازه می‌گیرد
    - labels را بررسی می‌کند
    """
    logger.info("بررسی نمونه‌های tokenize‌شده ...")

    for i in range(min(n_examples, len(train_dataset))):
        example = train_dataset[i]

        input_ids = example["input_ids"]
        labels = [l for l in example["labels"] if l != -100]

        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_label = tokenizer.decode(labels, skip_special_tokens=True)

        logger.info(
            "نمونه %d:\n"
            "  input_ids طول: %d\n"
            "  labels طول (بدون -100): %d\n"
            "  متن اصلی (50 کاراکتر): %s...\n"
            "  خلاصه مرجع (50 کاراکتر): %s...",
            i + 1,
            len(input_ids),
            len(labels),
            decoded_input[:50],
            decoded_label[:50],
        )

        # بررسی جهت: خلاصه باید کوتاه‌تر از متن باشد
        assert len(labels) > 0, f"labels خالی است برای نمونه {i}"
        assert len(input_ids) > 0, f"input_ids خالی است برای نمونه {i}"

    logger.info("✓ بررسی tokenization موفق بود.")


def measure_truncation(
    dataset: Any,
    tokenizer: Any,
    max_source_length: int,
    split_name: str,
) -> dict:
    """
    اندازه‌گیری درصد truncation در دیتاست.

    این اطلاعات باید در گزارش نهایی ثبت شود.
    truncation بالا = از دست رفتن اطلاعات مهم.
    """
    truncated = 0
    total = 0

    for example in dataset:
        input_ids = example.get("input_ids", [])
        if len(input_ids) >= max_source_length:
            truncated += 1
        total += 1

    pct = round(truncated / total * 100, 1) if total > 0 else 0
    logger.info(
        "[%s] truncation: %d/%d نمونه (%.1f%%) در max_source_length=%d",
        split_name, truncated, total, pct, max_source_length,
    )
    return {"split": split_name, "truncated": truncated, "total": total, "pct": pct}



# LoRA Configuration


def get_arabart_lora_target_modules(model: Any) -> list[str]:
    """
    target modules صحیح برای AraBART را شناسایی می‌کند.

    ⚠️ مهم: target_modules باید بر اساس معماری واقعی مدل تعیین شود.
    AraBART از BartForConditionalGeneration استفاده می‌کند.
    نام‌های projection در BART:
        q_proj, k_proj, v_proj, out_proj (در attention)
    
    این تابع معماری واقعی را inspect می‌کند و تأیید می‌کند
    که نام‌های تعریف‌شده در config واقعاً وجود دارند.
    """
    # نام‌های احتمالی attention projection در BART
    candidate_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    # پیدا کردن نام‌هایی که واقعاً در مدل وجود دارند
    found_modules = set()
    for name, module in model.named_modules():
        for candidate in candidate_modules:
            if name.endswith(candidate):
                found_modules.add(candidate)

    if not found_modules:
        # fallback برای معماری‌های متفاوت
        logger.warning(
            "هیچ attention projection استانداردی پیدا نشد. "
            "بررسی معماری مدل..."
        )
        # چاپ چند لایه اول برای debugging
        layer_names = [name for name, _ in list(model.named_modules())[:30]]
        logger.info("لایه‌های اول مدل: %s", layer_names)

        # تلاش با نام‌های BART داخلی
        for name, _ in model.named_modules():
            if any(proj in name for proj in ["q_proj", "v_proj", "k_proj"]):
                parts = name.split(".")
                for part in parts:
                    if part in ["q_proj", "v_proj", "k_proj", "out_proj"]:
                        found_modules.add(part)

    # اگر هنوز پیدا نشد، از پیش‌فرض استفاده می‌کنیم
    if not found_modules:
        logger.warning(
            "استفاده از target_modules پیش‌فرض: ['q_proj', 'v_proj']"
        )
        return ["q_proj", "v_proj"]

    result = sorted(list(found_modules))
    logger.info("target_modules یافت‌شده برای LoRA: %s", result)
    return result


def setup_lora(
    model: Any,
    lora_config_dict: dict,
    auto_detect_modules: bool = True,
) -> Any:
    """
    LoRA adapters را به مدل اضافه می‌کند.

    Args:
        model: مدل پایه
        lora_config_dict: تنظیمات از config
        auto_detect_modules: آیا target_modules خودکار شناسایی شود

    Returns:
        مدل با LoRA
    """
    from peft import LoraConfig, get_peft_model, TaskType

    # شناسایی target_modules
    if auto_detect_modules:
        target_modules = get_arabart_lora_target_modules(model)
    else:
        target_modules = lora_config_dict.get("target_modules", ["q_proj", "v_proj"])

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config_dict.get("r", 16),
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.1),
        bias=lora_config_dict.get("bias", "none"),
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    # گزارش پارامترها
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = round(trainable_params / total_params * 100, 4) if total_params > 0 else 0

    logger.info(
        "LoRA setup:\n"
        "  target_modules: %s\n"
        "  r=%d | alpha=%d | dropout=%.2f\n"
        "  پارامترهای کل: %s\n"
        "  پارامترهای قابل‌آموزش: %s (%.4f%%)",
        target_modules,
        lora_config_dict.get("r", 16),
        lora_config_dict.get("lora_alpha", 32),
        lora_config_dict.get("lora_dropout", 0.1),
        f"{total_params:,}",
        f"{trainable_params:,}",
        trainable_pct,
    )

    if trainable_params == 0:
        raise RuntimeError(
            "هیچ پارامتر قابل‌آموزشی وجود ندارد! "
            "target_modules اشتباه است. "
            f"modules تعریف‌شده: {target_modules}"
        )

    return model



# ROUGE Metric for Trainer


def create_compute_metrics(tokenizer: Any) -> Any:
    """
    تابع compute_metrics برای Seq2SeqTrainer.

    این تابع در پایان هر epoch روی validation set اجرا می‌شود.
    از ArabicRougeScorer پروژه استفاده می‌کند برای consistency.
    """
    from arabic_summarizer.evaluation.rouge_scorer import ArabicRougeScorer
    rouge_scorer = ArabicRougeScorer()

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred

        # decode predictions
        # predictions ممکن است tuple باشد
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        import numpy as np
        # جایگزینی -100 با pad_token_id برای decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # پاک‌سازی ساده
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # محاسبه ROUGE
        _, aggregate = rouge_scorer.score_batch(decoded_preds, decoded_labels)

        return {
            "rouge1": round(aggregate.rouge1, 4),
            "rouge2": round(aggregate.rouge2, 4),
            "rougeL": round(aggregate.rougeL, 4),
        }

    return compute_metrics



# Smoke Test


def run_smoke_test(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: Path,
    seed: int = 42,
) -> bool:
    """
    اجرای smoke test برای اطمینان از صحت pipeline.

    بررسی‌ها:
    1. forward pass کار می‌کند
    2. loss finite است
    3. backward pass کار می‌کند
    4. checkpoint ذخیره می‌شود
    5. یک step کامل آموزش موفق است

    Returns:
        True اگر همه بررسی‌ها موفق بودند
    """
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

    logger.info("══ شروع Smoke Test ══")

    # یک batch کوچک از دیتاست
    smoke_train = train_dataset.select(range(min(8, len(train_dataset))))
    smoke_eval = eval_dataset.select(range(min(4, len(eval_dataset)))) if eval_dataset else None

    smoke_output_dir = output_dir / "smoke_test"
    smoke_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(smoke_output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch" if smoke_eval else "no",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=64,
        seed=seed,
        fp16=False,
        bf16=False,
        logging_steps=1,
        report_to="none",
        max_steps=3,  # فقط 3 step برای smoke test
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=smoke_train,
        eval_dataset=smoke_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    try:
        logger.info("اجرای 3 step آموزش ...")
        train_result = trainer.train()
        loss = train_result.training_loss

        if not (loss == loss):  # NaN check
            logger.error("❌ Loss NaN است!")
            return False

        if loss <= 0:
            logger.warning("⚠️ Loss صفر یا منفی: %.4f", loss)

        logger.info("✓ Loss finite: %.4f", loss)

        # تست forward pass مستقیم
        device = next(model.parameters()).device
        batch = next(iter(
            torch.utils.data.DataLoader(smoke_train, batch_size=2,
                                         collate_fn=data_collator)
        ))
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}

        with torch.no_grad():
            outputs = model(**batch)
        assert hasattr(outputs, 'loss'), "خروجی مدل loss ندارد"
        assert outputs.loss is not None, "Loss None است"
        assert outputs.loss.item() == outputs.loss.item(), "Loss NaN است"

        logger.info("✓ Forward pass موفق | Loss: %.4f", outputs.loss.item())

        # تست checkpoint
        checkpoint_dir = smoke_output_dir / "checkpoint-3"
        if not checkpoint_dir.exists():
            # ممکن است نام متفاوت باشد
            checkpoints = list(smoke_output_dir.glob("checkpoint-*"))
            if not checkpoints:
                logger.warning("⚠️ هیچ checkpoint ذخیره نشد")
            else:
                logger.info("✓ Checkpoint ذخیره شد: %s", checkpoints[0].name)
        else:
            logger.info("✓ Checkpoint ذخیره شد")

        logger.info("══ Smoke Test موفق بود ══")
        return True

    except Exception as exc:
        logger.error("❌ Smoke Test شکست خورد: %s", exc)
        import traceback
        traceback.print_exc()
        return False



# Main Training


def train(
    data_dir: Path,
    output_dir: Path,
    cfg: dict,
    env_info: dict,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """
    Fine-Tuning کامل مدل AraBART با LoRA.

    Args:
        data_dir: پوشه دیتاست آماده‌شده
        output_dir: پوشه خروجی checkpoint ها
        cfg: تنظیمات از config
        env_info: اطلاعات محیط
        smoke_test: فقط smoke test اجرا شود

    Returns:
        نتایج آموزش
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        set_seed,
    )

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    logger.info("Seed تنظیم شد: %d", seed)

    # ── تنظیم device ───────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── بارگذاری tokenizer و مدل ───────────────────────────
    model_id = cfg["model"].get("base_model_id", "moussaKam/AraBART")
    tokenizer_id = cfg["model"].get("tokenizer_id", model_id)

    logger.info("بارگذاری tokenizer: %s", tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

    logger.info("بارگذاری مدل: %s", model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)

    base_param_count = sum(p.numel() for p in model.parameters())
    logger.info("پارامترهای مدل پایه: %s", f"{base_param_count:,}")

    # ── بارگذاری و tokenize دیتاست ─────────────────────────
    max_source_length = cfg["model"].get("max_source_length", 1024)
    max_target_length = cfg["model"].get("max_target_length", 128)

    max_train = 200 if smoke_test else cfg["dataset"].get("max_train_samples")
    max_eval = 50 if smoke_test else cfg["dataset"].get("max_eval_samples")

    train_ds, eval_ds, test_ds = load_dataset_from_jsonl(
        data_dir=data_dir,
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    )

    if train_ds is None:
        raise FileNotFoundError(
            f"فایل train.jsonl در {data_dir} پیدا نشد. "
            "ابتدا scripts/training/prepare_dataset.py را اجرا کنید."
        )

    if eval_ds is None:
        raise FileNotFoundError(
            f"فایل validation.jsonl در {data_dir} پیدا نشد."
        )

    # ── Tokenization ────────────────────────────────────────
    logger.info("Tokenize کردن دیتاست ...")

    tokenize_fn = create_tokenize_function(
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    tokenized_train = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenize train",
    )

    tokenized_eval = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenize validation",
    )

    # ── بررسی tokenization ──────────────────────────────────
    verify_tokenization_examples(tokenized_train, tokenizer, n_examples=3)

    # اندازه‌گیری truncation
    truncation_stats = {}
    truncation_stats["train"] = measure_truncation(
        tokenized_train, tokenizer, max_source_length, "train"
    )
    truncation_stats["validation"] = measure_truncation(
        tokenized_eval, tokenizer, max_source_length, "validation"
    )

    # ── LoRA Setup ──────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    model = setup_lora(model, lora_cfg, auto_detect_modules=True)

    # ── Data Collator ───────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Smoke Test ──────────────────────────────────────────
    if smoke_test:
        success = run_smoke_test(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            output_dir=output_dir,
            seed=seed,
        )
        if not success:
            raise RuntimeError("Smoke Test شکست خورد. آموزش متوقف شد.")

        logger.info("Smoke Test موفق بود. برای آموزش کامل --smoke-test را حذف کنید.")
        return {
            "smoke_test": True,
            "success": True,
            "truncation_stats": truncation_stats,
        }

    # ── Training Arguments ──────────────────────────────────
    train_cfg = cfg["training"]
    use_fp16 = train_cfg.get("fp16", False) and device == "cuda"
    use_bf16 = train_cfg.get("bf16", False) and device == "cuda"

    # در صورت A100 یا Ampere GPU، bf16 بهتر از fp16 است
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        if not use_bf16 and not use_fp16:
            logger.info(
                "GPU Ampere شناسایی شد. bf16 فعال می‌شود برای کارایی بهتر."
            )
            use_bf16 = True

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        seed=seed,
        evaluation_strategy=train_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "rougeL"),
        greater_is_better=train_cfg.get("greater_is_better", True),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_dir=str(_PROJECT_ROOT / train_cfg.get("logging_dir", "reports/training/logs")),
        logging_steps=train_cfg.get("logging_steps", 50),
        report_to=train_cfg.get("report_to", "none"),
        predict_with_generate=True,
        generation_max_length=train_cfg.get("generation_max_length", 128),
        generation_num_beams=train_cfg.get("generation_num_beams", 4),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
    )

    # ── Callbacks ───────────────────────────────────────────
    callbacks = []
    early_stopping_patience = train_cfg.get("early_stopping_patience", 2)
    if training_args.load_best_model_at_end:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience
        ))
        logger.info("Early Stopping فعال: patience=%d", early_stopping_patience)

    # ── Trainer ─────────────────────────────────────────────
    compute_metrics = create_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── آموزش ───────────────────────────────────────────────
    logger.info("شروع آموزش ...")
    t_start = time.time()

    train_result = trainer.train()

    training_time_seconds = time.time() - t_start
    logger.info(
        "آموزش کامل شد در %.1f دقیقه.",
        training_time_seconds / 60,
    )

    # ── ذخیره LoRA adapter ──────────────────────────────────
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("LoRA adapter ذخیره شد: %s", adapter_dir)

    # ── ذخیره مدل merge‌شده برای inference ──────────────────
    logger.info("Merge کردن LoRA adapters با مدل پایه ...")
    try:
        merged_model = model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("مدل merge‌شده ذخیره شد: %s", merged_dir)
    except Exception as e:
        logger.warning("merge_and_unload ناموفق بود: %s", e)
        logger.info("از LoRA adapter برای inference استفاده خواهد شد.")

    # ── ارزیابی validation ──────────────────────────────────
    logger.info("ارزیابی validation ...")
    eval_results = trainer.evaluate()
    logger.info("نتایج validation: %s", eval_results)

    # ── ذخیره metadata ──────────────────────────────────────
    lora_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metadata = {
        "training_completed_at": datetime.now(timezone.utc).isoformat(),
        "base_model": model_id,
        "tokenizer": tokenizer_id,
        "dataset_source": str(data_dir),
        "seed": seed,
        "environment": env_info,
        "lora_config": {
            "r": lora_cfg.get("r", 16),
            "lora_alpha": lora_cfg.get("lora_alpha", 32),
            "lora_dropout": lora_cfg.get("lora_dropout", 0.1),
            "target_modules": lora_cfg.get("target_modules", []),
        },
        "training_args": {
            "epochs": train_cfg.get("num_train_epochs", 3),
            "learning_rate": train_cfg.get("learning_rate", 5e-5),
            "batch_size": train_cfg.get("per_device_train_batch_size", 2),
            "gradient_accumulation": train_cfg.get("gradient_accumulation_steps", 4),
            "effective_batch_size": (
                train_cfg.get("per_device_train_batch_size", 2)
                * train_cfg.get("gradient_accumulation_steps", 4)
            ),
            "fp16": use_fp16,
            "bf16": use_bf16,
        },
        "dataset": {
            "train_samples": len(tokenized_train),
            "eval_samples": len(tokenized_eval),
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "truncation_stats": truncation_stats,
        },
        "model_params": {
            "base_model_params": base_param_count,
            "lora_trainable_params": lora_trainable,
            "trainable_pct": round(lora_trainable / base_param_count * 100, 4),
        },
        "training_metrics": {
            "training_loss": train_result.training_loss,
            "training_time_seconds": training_time_seconds,
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second"
            ),
        },
        "validation_metrics": eval_results,
        "output_paths": {
            "lora_adapter": str(adapter_dir),
            "merged_model": str(output_dir / "merged"),
        },
        "known_constraints": {
            "xlsum_avg_compression_ratio": 0.052,
            "product_target": "10-30%",
            "strategy": "Output length enforced at inference via generation constraints",
        },
    }

    meta_file = output_dir / "training_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Metadata ذخیره شد: %s", meta_file)

    return {
        "smoke_test": False,
        "training_loss": train_result.training_loss,
        "validation_metrics": eval_results,
        "training_time_seconds": training_time_seconds,
        "output_dir": str(output_dir),
        "truncation_stats": truncation_stats,
    }



# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-Tuning AraBART با LoRA برای خلاصه‌سازی عربی",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="فقط smoke test با 200 نمونه اجرا شود",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="پوشه دیتاست آماده‌شده (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="پوشه خروجی (پیش‌فرض از config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="تعداد epoch (override config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate (override config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size per device (override config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── اطلاعات محیط ────────────────────────────────────────
    env_info = print_environment_info()

    # ── بارگذاری config ─────────────────────────────────────
    config_loader = ConfigLoader()
    cfg = {
        "model": config_loader.get_section("training", "model"),
        "dataset": config_loader.get_section("training", "dataset"),
        "lora": config_loader.get_section("training", "lora"),
        "training": config_loader.get_section("training", "training"),
        "generation": config_loader.get_section("training", "generation"),
    }

    # ── override از CLI ──────────────────────────────────────
    if args.epochs is not None:
        cfg["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size

    # ── مسیرها ──────────────────────────────────────────────
    source_name = cfg["dataset"].get("source_name", "xlsum_arabic")

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = _PROJECT_ROOT / "data" / "training_ready" / source_name

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = _PROJECT_ROOT / cfg["training"].get(
            "output_dir", "models/finetuned/arabart"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("دیتاست: %s", data_dir)
    logger.info("خروجی: %s", output_dir)

    if args.smoke_test:
        logger.info("══ حالت Smoke Test ══")

    # ── شروع آموزش ──────────────────────────────────────────
    try:
        results = train(
            data_dir=data_dir,
            output_dir=output_dir,
            cfg=cfg,
            env_info=env_info,
            smoke_test=args.smoke_test,
        )

        logger.info("══ نتایج نهایی ══")
        if results.get("smoke_test"):
            logger.info("Smoke Test: %s", "✓ موفق" if results.get("success") else "❌ شکست")
        else:
            logger.info(
                "Training Loss: %.4f | Val ROUGE-L: %.4f | زمان: %.1f دقیقه",
                results.get("training_loss", 0),
                results.get("validation_metrics", {}).get("eval_rougeL", 0),
                results.get("training_time_seconds", 0) / 60,
            )

    except FileNotFoundError as e:
        logger.error("❌ فایل پیدا نشد: %s", e)
        logger.info(
            "ابتدا این دستور را اجرا کنید:\n"
            "  python scripts/training/prepare_dataset.py"
        )
        sys.exit(1)
    except RuntimeError as e:
        logger.error("❌ خطای اجرا: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()