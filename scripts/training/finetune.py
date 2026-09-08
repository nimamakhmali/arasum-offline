"""
TASK-202: Fine-Tuning مدل AraBART با LoRA.

اصلاحات این نسخه:
1. رفع OverflowError در compute_metrics
2. پشتیبانی از resume_from_checkpoint
3. تشخیص خودکار GPU و تنظیم precision
4. مسیر dataset اصلاح شد
5. evaluation_strategy به steps تغییر کرد

اجرا:
    python scripts/training/finetune.py --smoke-test
    python scripts/training/finetune.py
    python scripts/training/finetune.py --resume-from-checkpoint models/finetuned/arabart/checkpoint-500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from arabic_summarizer.utils.config_loader import ConfigLoader
from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════

def print_environment_info() -> dict[str, Any]:
    """اطلاعات محیط را چاپ و برمی‌گرداند."""
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
            info["gpu_capability"] = torch.cuda.get_device_capability()
            logger.info(
                "GPU: %s | VRAM: %.1f GB | Capability: %s",
                info["gpu_name"],
                info["gpu_vram_gb"],
                info["gpu_capability"],
            )
        else:
            logger.warning("CUDA موجود نیست. آموزش روی CPU بسیار کند خواهد بود.")
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
        logger.error("PEFT نصب نیست!")
        raise

    logger.info(
        "محیط: Python=%s | PyTorch=%s | Transformers=%s | PEFT=%s",
        info["python_version"],
        info.get("torch_version", "?"),
        info.get("transformers_version", "?"),
        info.get("peft_version", "?"),
    )
    return info


def detect_precision(device: str) -> tuple[bool, bool]:
    """
    بهترین precision را بر اساس GPU تشخیص می‌دهد.

    Returns:
        (use_fp16, use_bf16)
    """
    import torch

    if device != "cuda":
        return False, False

    capability = torch.cuda.get_device_capability()
    major = capability[0]

    if major >= 8:
        # Ampere یا جدیدتر (A100، RTX 3000+، RTX 4000+)
        logger.info("GPU Ampere+ شناسایی شد → BF16 فعال")
        return False, True
    else:
        # قدیمی‌تر (P100، V100، T4)
        logger.info("GPU قدیمی‌تر شناسایی شد → FP16 فعال")
        return True, False


def detect_batch_size(device: str, base_batch: int = 4) -> tuple[int, int]:
    """
    batch size مناسب را بر اساس VRAM تشخیص می‌دهد.

    Returns:
        (per_device_batch_size, gradient_accumulation_steps)
    """
    import torch

    if device != "cuda":
        return 2, 4  # CPU: کوچک‌ترین batch

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if vram_gb >= 40:      # A100 80GB / A100 40GB
        batch, accum = 16, 1
    elif vram_gb >= 16:    # V100 / A10
        batch, accum = 8, 2
    elif vram_gb >= 12:    # RTX 3080 Ti / Tesla T4
        batch, accum = 4, 2
    elif vram_gb >= 8:     # RTX 3070 / P100
        batch, accum = 2, 4
    else:
        batch, accum = 2, 4

    effective = batch * accum
    logger.info(
        "VRAM=%.1fGB → batch=%d × accum=%d = effective=%d",
        vram_gb, batch, accum, effective,
    )
    return batch, accum


# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════

def load_dataset_from_jsonl(
    data_dir: Path,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> tuple[Any, Any, Any]:
    """دیتاست را از فایل‌های JSONL می‌خواند."""
    from datasets import Dataset

    def read_jsonl(file_path: Path, max_samples: Optional[int]) -> Optional[Any]:
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
                        rec = json.loads(line)
                        if rec.get("text") and rec.get("summary"):
                            records.append(rec)
                    except json.JSONDecodeError:
                        pass

        if not records:
            return None

        logger.info(
            "%s: %d رکورد بارگذاری شد", file_path.name, len(records)
        )
        return Dataset.from_list(records)

    train_ds = read_jsonl(data_dir / "train.jsonl", max_train_samples)
    eval_ds = read_jsonl(data_dir / "validation.jsonl", max_eval_samples)
    test_ds = read_jsonl(data_dir / "test.jsonl", None)

    logger.info(
        "دیتاست: train=%s | val=%s | test=%s",
        len(train_ds) if train_ds else "N/A",
        len(eval_ds) if eval_ds else "N/A",
        len(test_ds) if test_ds else "N/A",
    )
    return train_ds, eval_ds, test_ds


# ══════════════════════════════════════════════════════════
# Tokenization
# ══════════════════════════════════════════════════════════

def create_tokenize_function(
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
):
    """
    تابع tokenization اصلاح‌شده.
    از text_target به جای as_target_tokenizer استفاده می‌کند.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def tokenize_function(examples: dict) -> dict:
        # encode source
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )

        # encode target — روش جدید (بدون as_target_tokenizer deprecated)
        label_encoding = tokenizer(
            text_target=examples["summary"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )

        # جایگزینی pad_token_id با -100
        model_inputs["labels"] = [
            [token if token != pad_id else -100 for token in ids]
            for ids in label_encoding["input_ids"]
        ]

        return model_inputs

    return tokenize_function


def verify_tokenization(train_dataset: Any, tokenizer: Any, n: int = 3) -> None:
    """چند نمونه tokenize‌شده را بررسی می‌کند."""
    logger.info("بررسی نمونه‌های tokenize‌شده ...")
    for i in range(min(n, len(train_dataset))):
        ex = train_dataset[i]
        input_ids = ex["input_ids"]
        labels = [l for l in ex["labels"] if l != -100]
        inp_text = tokenizer.decode(input_ids[:30], skip_special_tokens=True)
        lab_text = tokenizer.decode(labels[:20], skip_special_tokens=True)
        logger.info(
            "نمونه %d: input=%d tokens | label=%d tokens\n"
            "  input: %s...\n"
            "  label: %s...",
            i + 1, len(input_ids), len(labels),
            inp_text[:60], lab_text[:60],
        )
    logger.info("✓ بررسی tokenization موفق")


def measure_truncation(dataset: Any, max_length: int, split_name: str) -> dict:
    """اندازه‌گیری درصد truncation."""
    truncated = sum(
        1 for ex in dataset if len(ex.get("input_ids", [])) >= max_length
    )
    total = len(dataset)
    pct = round(truncated / total * 100, 1) if total > 0 else 0
    logger.info(
        "[%s] truncation: %d/%d (%.1f%%) در max=%d",
        split_name, truncated, total, pct, max_length,
    )
    return {"split": split_name, "truncated": truncated, "total": total, "pct": pct}


# ══════════════════════════════════════════════════════════
# LoRA
# ══════════════════════════════════════════════════════════

def get_target_modules(model: Any) -> list[str]:
    """target modules واقعی مدل را شناسایی می‌کند."""
    candidates = {"q_proj", "v_proj", "k_proj", "out_proj"}
    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                found.add(c)
    if not found:
        logger.warning("هیچ attention module پیدا نشد. از پیش‌فرض استفاده می‌شود.")
        return ["q_proj", "v_proj"]
    result = sorted(list(found))
    logger.info("target_modules: %s", result)
    return result


def setup_lora(model: Any, lora_cfg: dict) -> Any:
    """LoRA adapters را اضافه می‌کند."""
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = get_target_modules(model)

    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
    )

    model = get_peft_model(model, config)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = round(trainable / total * 100, 4) if total > 0 else 0

    logger.info(
        "LoRA: r=%d | alpha=%d | modules=%s\n"
        "  کل: %s | قابل‌آموزش: %s (%.4f%%)",
        lora_cfg.get("r", 16), lora_cfg.get("lora_alpha", 32),
        target_modules, f"{total:,}", f"{trainable:,}", pct,
    )

    if trainable == 0:
        raise RuntimeError("هیچ پارامتر قابل‌آموزشی وجود ندارد!")

    return model, target_modules


# ══════════════════════════════════════════════════════════
# Compute Metrics — نسخه اصلاح‌شده
# ══════════════════════════════════════════════════════════

def create_compute_metrics(tokenizer: Any) -> Any:
    """
    compute_metrics اصلاح‌شده.

    اصلاح OverflowError:
    - predictions کلیپ می‌شوند به [0, vocab_size-1]
    - هر token به صورت جداگانه decode می‌شود با error handling
    - جفت‌های خالی فیلتر می‌شوند
    """
    import re
    from rouge_score import rouge_scorer as rs

    class ArabicWhitespaceTok:
        def tokenize(self, text: str) -> list[str]:
            text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
            return [t for t in text.split() if t]

    scorer = rs.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=ArabicWhitespaceTok(),
    )

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    def compute_metrics(eval_pred) -> dict[str, float]:
        import numpy as np

        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # ── کلیپ کردن برای جلوگیری از OverflowError ──────────
        predictions = np.clip(
            predictions.astype(np.int64), 0, vocab_size - 1
        ).astype(np.int32)

        labels_fixed = np.where(labels != -100, labels, pad_id)
        labels_fixed = np.clip(
            labels_fixed.astype(np.int64), 0, vocab_size - 1
        ).astype(np.int32)

        # ── Decode با error handling کامل ─────────────────────
        decoded_preds, decoded_refs = [], []

        for pred_ids, label_ids in zip(predictions, labels_fixed):
            try:
                pred_list = [int(x) for x in pred_ids
                             if 0 <= int(x) < vocab_size]
                pred_text = tokenizer.decode(
                    pred_list,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
            except Exception:
                pred_text = ""

            try:
                label_list = [int(x) for x in label_ids
                              if 0 <= int(x) < vocab_size]
                label_text = tokenizer.decode(
                    label_list,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
            except Exception:
                label_text = ""

            decoded_preds.append(pred_text)
            decoded_refs.append(label_text)

        # ── محاسبه ROUGE ───────────────────────────────────────
        r1, r2, rL = [], [], []
        for p, r in zip(decoded_preds, decoded_refs):
            if p and r:
                try:
                    s = scorer.score(r, p)
                    r1.append(s["rouge1"].fmeasure)
                    r2.append(s["rouge2"].fmeasure)
                    rL.append(s["rougeL"].fmeasure)
                except Exception:
                    pass

        if not r1:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        return {
            "rouge1": round(sum(r1) / len(r1), 4),
            "rouge2": round(sum(r2) / len(r2), 4),
            "rougeL": round(sum(rL) / len(rL), 4),
        }

    return compute_metrics


# ══════════════════════════════════════════════════════════
# Smoke Test
# ══════════════════════════════════════════════════════════

def run_smoke_test(
    model: Any, tokenizer: Any,
    train_dataset: Any, eval_dataset: Any,
    output_dir: Path, device: str, seed: int,
) -> bool:
    """Smoke test با 3 step."""
    import torch
    from transformers import (
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )

    logger.info("══ Smoke Test ══")
    smoke_train = train_dataset.select(range(min(8, len(train_dataset))))
    smoke_eval = eval_dataset.select(range(min(4, len(eval_dataset)))) \
        if eval_dataset else None

    smoke_dir = output_dir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    use_fp16, use_bf16 = detect_precision(device)

    args = Seq2SeqTrainingArguments(
        output_dir=str(smoke_dir),
        max_steps=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps" if smoke_eval else "no",
        eval_steps=3,
        save_strategy="steps",
        save_steps=3,
        predict_with_generate=True,
        generation_max_length=32,
        generation_num_beams=2,
        seed=seed,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=1,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=smoke_train,
        eval_dataset=smoke_eval,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=create_compute_metrics(tokenizer),
    )

    try:
        result = trainer.train()
        loss = result.training_loss
        if loss != loss:  # NaN
            logger.error("❌ Loss NaN!")
            return False
        logger.info("✓ Loss: %.4f", loss)

        # تست forward مستقیم
        batch = next(iter(
            torch.utils.data.DataLoader(smoke_train, batch_size=2, collate_fn=collator)
        ))
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = model(**batch)
        assert out.loss.item() == out.loss.item(), "Loss NaN در forward"
        logger.info("✓ Forward pass: Loss=%.4f", out.loss.item())

        # بررسی checkpoint
        checkpoints = list(smoke_dir.glob("checkpoint-*"))
        if checkpoints:
            logger.info("✓ Checkpoint: %s", checkpoints[0].name)
        else:
            logger.warning("⚠️ هیچ checkpoint ذخیره نشد")

        logger.info("══ Smoke Test موفق ══")
        return True

    except Exception as e:
        import traceback
        logger.error("❌ Smoke Test شکست: %s", e)
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════
# Main Training
# ══════════════════════════════════════════════════════════

def train(
    data_dir: Path,
    output_dir: Path,
    cfg: dict,
    env_info: dict,
    smoke_test: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Fine-Tuning کامل با پشتیبانی از resume."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
        set_seed,
    )

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Precision تشخیص خودکار ──────────────────────────────
    use_fp16, use_bf16 = detect_precision(device)

    # ── Batch size تشخیص خودکار ─────────────────────────────
    auto_batch, auto_accum = detect_batch_size(device)

    # Config override با auto-detection
    train_cfg = cfg["training"]
    batch_size = train_cfg.get("per_device_train_batch_size", auto_batch)
    grad_accum = train_cfg.get("gradient_accumulation_steps", auto_accum)

    # اگر CPU است، کوچک‌تر کن
    if device == "cpu":
        batch_size = 2
        grad_accum = 4

    logger.info(
        "Training config: batch=%d × accum=%d = effective=%d",
        batch_size, grad_accum, batch_size * grad_accum,
    )

    # ── بارگذاری tokenizer و مدل ────────────────────────────
    model_id = cfg["model"].get("base_model_id", "moussaKam/AraBART")
    tokenizer_id = cfg["model"].get("tokenizer_id", model_id)
    max_source_length = cfg["model"].get("max_source_length", 1024)
    max_target_length = cfg["model"].get("max_target_length", 128)

    logger.info("بارگذاری tokenizer: %s", tokenizer_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

    logger.info("بارگذاری مدل: %s", model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)

    base_params = sum(p.numel() for p in model.parameters())
    logger.info("پارامترهای پایه: %s", f"{base_params:,}")

    # ── دیتاست ──────────────────────────────────────────────
    max_train = 200 if smoke_test else cfg["dataset"].get("max_train_samples")
    max_eval = 50 if smoke_test else cfg["dataset"].get("max_eval_samples")

    train_ds, eval_ds, test_ds = load_dataset_from_jsonl(
        data_dir=data_dir,
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    )

    if train_ds is None:
        raise FileNotFoundError(
            f"train.jsonl در {data_dir} پیدا نشد.\n"
            "اجرا کنید: python scripts/training/prepare_dataset.py"
        )
    if eval_ds is None:
        raise FileNotFoundError(f"validation.jsonl در {data_dir} پیدا نشد.")

    # ── Tokenization ────────────────────────────────────────
    logger.info("Tokenize کردن ...")
    tokenize_fn = create_tokenize_function(tokenizer, max_source_length, max_target_length)

    tokenized_train = train_ds.map(
        tokenize_fn, batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenize train",
    )
    tokenized_eval = eval_ds.map(
        tokenize_fn, batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenize val",
    )

    verify_tokenization(tokenized_train, tokenizer)

    trunc_stats = {
        "train": measure_truncation(tokenized_train, max_source_length, "train"),
        "validation": measure_truncation(tokenized_eval, max_source_length, "validation"),
    }

    # ── LoRA ─────────────────────────────────────────────────
    model, target_modules = setup_lora(model, cfg.get("lora", {}))

    # ── Smoke Test ───────────────────────────────────────────
    if smoke_test:
        success = run_smoke_test(
            model=model, tokenizer=tokenizer,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            output_dir=output_dir,
            device=device, seed=seed,
        )
        if not success:
            raise RuntimeError("Smoke Test شکست خورد.")
        logger.info("Smoke Test موفق. --smoke-test را حذف کنید برای آموزش کامل.")
        return {"smoke_test": True, "success": True}

    # ── Data Collator ────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Training Arguments ───────────────────────────────────
    eval_steps = train_cfg.get("eval_steps", 500)
    save_steps = train_cfg.get("save_steps", 500)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        seed=seed,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_dir=str(_PROJECT_ROOT / train_cfg.get("logging_dir", "reports/training/logs")),
        logging_steps=train_cfg.get("logging_steps", 100),
        report_to="none",
        predict_with_generate=True,
        generation_max_length=train_cfg.get("generation_max_length", 128),
        generation_num_beams=train_cfg.get("generation_num_beams", 4),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
    )

    # ── Trainer ──────────────────────────────────────────────
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=train_cfg.get("early_stopping_patience", 2)
        )
    ]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=create_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    # ── آموزش با پشتیبانی از Resume ─────────────────────────
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            # جستجوی آخرین checkpoint
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda d: int(d.name.split("-")[-1]),
                reverse=True,
            )
            if checkpoints:
                resume_from_checkpoint = str(checkpoints[0])
                logger.info("Resume از: %s", resume_from_checkpoint)
            else:
                logger.warning("Checkpoint پیدا نشد. از ابتدا شروع می‌شود.")
                resume_from_checkpoint = None

    logger.info("شروع آموزش ...")
    t_start = time.time()

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    duration = time.time() - t_start
    logger.info("آموزش کامل شد در %.1f دقیقه", duration / 60)

    # ── ذخیره مدل ───────────────────────────────────────────
    _save_model(model, tokenizer, output_dir, base_params, target_modules, cfg)

    # ── ارزیابی validation ───────────────────────────────────
    logger.info("ارزیابی validation ...")
    eval_results = trainer.evaluate()
    logger.info("نتایج: %s", eval_results)

    # ── metadata ─────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_model": model_id,
        "seed": seed,
        "environment": env_info,
        "lora_config": {
            "r": cfg.get("lora", {}).get("r", 16),
            "lora_alpha": cfg.get("lora", {}).get("lora_alpha", 32),
            "target_modules": target_modules,
        },
        "training_config": {
            "epochs": train_cfg.get("num_train_epochs", 3),
            "lr": train_cfg.get("learning_rate", 5e-5),
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "fp16": use_fp16,
            "bf16": use_bf16,
        },
        "dataset": {
            "train": len(tokenized_train),
            "validation": len(tokenized_eval),
            "truncation": trunc_stats,
        },
        "model_params": {
            "base": base_params,
            "trainable": trainable,
            "pct": round(trainable / (base_params + trainable) * 100, 4),
        },
        "results": {
            "training_loss": train_result.training_loss,
            "training_time_minutes": round(duration / 60, 1),
            "validation": eval_results,
        },
    }

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "smoke_test": False,
        "training_loss": train_result.training_loss,
        "validation_metrics": eval_results,
        "training_time_minutes": round(duration / 60, 1),
        "output_dir": str(output_dir),
    }


def _save_model(
    model: Any, tokenizer: Any, output_dir: Path,
    base_params: int, target_modules: list, cfg: dict,
) -> None:
    """مدل را ذخیره می‌کند."""
    # LoRA adapter
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("LoRA adapter: %s", adapter_dir)

    # مدل merge‌شده
    try:
        logger.info("Merge کردن LoRA ...")
        merged = model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("مدل merge‌شده: %s", merged_dir)
    except Exception as e:
        logger.warning("merge ناموفق: %s", e)


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-Tuning AraBART با LoRA"
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str, default=None,
        help="مسیر checkpoint برای ادامه آموزش (یا 'auto' برای آخرین)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_info = print_environment_info()

    cfg_loader = ConfigLoader()
    cfg = {
        "model": cfg_loader.get_section("training", "model"),
        "dataset": cfg_loader.get_section("training", "dataset"),
        "lora": cfg_loader.get_section("training", "lora"),
        "training": cfg_loader.get_section("training", "training"),
        "generation": cfg_loader.get_section("training", "generation"),
    }

    if args.epochs:
        cfg["training"]["num_train_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    source = cfg["dataset"].get("source_name", "xlsum_arabic")
    processed_dir = cfg["dataset"].get("processed_dir", "data/training_ready")

    data_dir = args.data_dir or (_PROJECT_ROOT / processed_dir / source)
    output_dir = args.output_dir or (
        _PROJECT_ROOT / cfg["training"].get("output_dir", "models/finetuned/arabart")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = args.resume_from_checkpoint
    if resume == "auto":
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda d: int(d.name.split("-")[-1]),
            reverse=True,
        )
        resume = str(checkpoints[0]) if checkpoints else None
        if resume:
            logger.info("Auto-resume از: %s", resume)

    logger.info("Dataset: %s", data_dir)
    logger.info("Output: %s", output_dir)

    try:
        results = train(
            data_dir=data_dir,
            output_dir=output_dir,
            cfg=cfg,
            env_info=env_info,
            smoke_test=args.smoke_test,
            resume_from_checkpoint=resume,
        )

        if results.get("smoke_test"):
            logger.info("Smoke Test: %s", "✓" if results.get("success") else "❌")
        else:
            logger.info(
                "نتیجه: Loss=%.4f | ROUGE-L=%.4f | زمان=%.1f دقیقه",
                results.get("training_loss", 0),
                results.get("validation_metrics", {}).get("eval_rougeL", 0),
                results.get("training_time_minutes", 0),
            )

    except FileNotFoundError as e:
        logger.error("❌ %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("❌ %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()