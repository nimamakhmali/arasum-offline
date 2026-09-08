"""
scripts/training/colab_finetune_complete.py

فایل کامل Fine-Tuning برای Google Colab.
تمام وابستگی‌های داخلی پروژه اینجا inline شده‌اند.

اجرا در Colab:
    1. Runtime > Change runtime type > T4 GPU
    2. این فایل را آپلود کنید
    3. python colab_finetune_complete.py

یا به صورت notebook:
    هر بخش # ══ CELL X ══ را در یک سلول جداگانه اجرا کنید
"""

from __future__ import annotations

# ══ CELL 1: نصب ══════════════════════════════════════════════
import subprocess, sys

def install():
    packages = [
        "transformers==4.41.2",
        "peft==0.11.1",
        "accelerate==0.31.0",
        "datasets==2.19.2",
        "rouge-score==0.1.2",
        "bert-score==0.3.13",
        "sentencepiece==0.2.0",
        "protobuf==4.25.3",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    print("✓ نصب کامل شد")

# install()  # فقط در Colab uncomment کنید


# ══ CELL 2: بررسی محیط ═══════════════════════════════════════
import torch, os, json, time, re, statistics
from pathlib import Path

def check_environment():
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        print(f"✓ GPU: {info['gpu']} | VRAM: {info['vram_gb']} GB")
    else:
        print("⚠️  CPU mode — بسیار کند!")
    
    import transformers, peft
    info["transformers"] = transformers.__version__
    info["peft"] = peft.__version__
    print(f"✓ PyTorch={info['torch']} | Transformers={info['transformers']}")
    return info


# ══ CELL 3: تنظیمات ══════════════════════════════════════════

# ── مسیرها ──────────────────────────────────────────────────
# در Colab با Google Drive:
USE_DRIVE = False  # True کنید اگر Drive mount شده

if USE_DRIVE:
    DATA_DIR = Path("/content/drive/MyDrive/arasum/data/training_ready/xlsum_arabic")
    OUTPUT_DIR = Path("/content/drive/MyDrive/arasum/models/arabart")
else:
    DATA_DIR = Path("/content/data/training_ready/xlsum_arabic")
    OUTPUT_DIR = Path("/content/models/arabart")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── تنظیمات مدل ─────────────────────────────────────────────
MODEL_ID = "moussaKam/AraBART"
TOKENIZER_ID = "moussaKam/AraBART"
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 128
SEED = 42

# ── تنظیمات LoRA ────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]

# ── تنظیمات آموزش ───────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# batch size بر اساس VRAM
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram >= 15:    # A100/V100
        BATCH_SIZE = 8
        GRAD_ACCUM = 2
    elif vram >= 12:  # T4/P100
        BATCH_SIZE = 4
        GRAD_ACCUM = 4
    else:             # کمتر از 12GB
        BATCH_SIZE = 2
        GRAD_ACCUM = 8
else:
    BATCH_SIZE = 2
    GRAD_ACCUM = 4

EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
EVAL_STEPS = 500  # ارزیابی هر 500 step
SAVE_STEPS = 500
LOG_STEPS = 100

print(f"Device: {DEVICE}")
print(f"Batch: {BATCH_SIZE} × accum {GRAD_ACCUM} = effective {EFFECTIVE_BATCH}")
print(f"Output: {OUTPUT_DIR}")


# ══ CELL 4: بارگذاری دیتاست ══════════════════════════════════

def load_jsonl(path: Path, max_samples=None):
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if rec.get("text") and rec.get("summary"):
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    return records

def load_all_splits(data_dir: Path):
    from datasets import Dataset
    splits = {}
    for split_name in ["train", "validation", "test"]:
        file_path = data_dir / f"{split_name}.jsonl"
        if file_path.exists():
            records = load_jsonl(file_path)
            splits[split_name] = Dataset.from_list(records)
            print(f"✓ {split_name}: {len(splits[split_name])} نمونه")
        else:
            print(f"⚠️  {split_name} پیدا نشد: {file_path}")
    return splits


# ══ CELL 5: Tokenization ══════════════════════════════════════

def get_tokenize_fn(tokenizer, max_src, max_tgt):
    """
    تابع tokenization اصلاح‌شده برای AraBART.
    از text_target به جای as_target_tokenizer استفاده می‌کند
    (روش جدید transformers >= 4.30).
    """
    pad_id = tokenizer.pad_token_id or 1

    def tokenize_fn(examples):
        # encode source
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_src,
            truncation=True,
            padding=False,
        )

        # encode target با روش جدید (بدون as_target_tokenizer)
        label_encoding = tokenizer(
            text_target=examples["summary"],
            max_length=max_tgt,
            truncation=True,
            padding=False,
        )

        # جایگزینی pad_token_id با -100
        model_inputs["labels"] = [
            [token if token != pad_id else -100 for token in label_ids]
            for label_ids in label_encoding["input_ids"]
        ]

        return model_inputs

    return tokenize_fn


# ══ CELL 6: ROUGE metric ══════════════════════════════════════

def build_compute_metrics(tokenizer):
    """
    compute_metrics اصلاح‌شده برای جلوگیری از OverflowError.
    """
    from rouge_score import rouge_scorer as rs

    class WhitespaceTok:
        def tokenize(self, text):
            text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
            return [t for t in text.split() if t]

    scorer = rs.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=WhitespaceTok(),
    )

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id or 1

    def compute_metrics(eval_pred):
        import numpy as np
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # ── کلیپ کردن predictions برای جلوگیری از OverflowError ──
        predictions = np.clip(
            predictions.astype(np.int64), 0, vocab_size - 1
        ).astype(np.int32)

        labels = np.where(labels != -100, labels, pad_id)
        labels = np.clip(labels.astype(np.int64), 0, vocab_size - 1).astype(np.int32)

        decoded_preds, decoded_refs = [], []

        for pred_ids, label_ids in zip(predictions, labels):
            try:
                p = tokenizer.decode(
                    pred_ids.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                decoded_preds.append(p)
            except Exception:
                decoded_preds.append("")

            try:
                r = tokenizer.decode(
                    label_ids.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                decoded_refs.append(r)
            except Exception:
                decoded_refs.append("")

        r1_scores, r2_scores, rL_scores = [], [], []
        for p, r in zip(decoded_preds, decoded_refs):
            if p and r:
                try:
                    s = scorer.score(r, p)
                    r1_scores.append(s["rouge1"].fmeasure)
                    r2_scores.append(s["rouge2"].fmeasure)
                    rL_scores.append(s["rougeL"].fmeasure)
                except Exception:
                    pass

        if not r1_scores:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        return {
            "rouge1": round(sum(r1_scores) / len(r1_scores), 4),
            "rouge2": round(sum(r2_scores) / len(r2_scores), 4),
            "rougeL": round(sum(rL_scores) / len(rL_scores), 4),
        }

    return compute_metrics


# ══ CELL 7: آموزش ════════════════════════════════════════════

def run_training(smoke_test: bool = False):
    import transformers
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    set_seed(SEED)
    env_info = check_environment()

    # ── بارگذاری دیتاست ─────────────────────────────────────
    splits = load_all_splits(DATA_DIR)
    if "train" not in splits:
        raise FileNotFoundError(f"train.jsonl پیدا نشد در {DATA_DIR}")

    max_train = 200 if smoke_test else None
    max_eval = 50 if smoke_test else None

    train_ds = splits["train"].select(
        range(min(max_train or len(splits["train"]), len(splits["train"])))
    )
    eval_ds = splits.get("validation")
    if eval_ds and max_eval:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    print(f"\nTrain: {len(train_ds)} | Val: {len(eval_ds) if eval_ds else 'N/A'}")

    # ── tokenizer و مدل ──────────────────────────────────────
    print("\nبارگذاری tokenizer و مدل ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.to(DEVICE)

    base_params = sum(p.numel() for p in model.parameters())
    print(f"پارامترهای پایه: {base_params:,}")

    # ── Tokenization ─────────────────────────────────────────
    print("\nTokenize کردن ...")
    tokenize_fn = get_tokenize_fn(tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

    tokenized_train = train_ds.map(
        tokenize_fn, batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenize train",
    )
    tokenized_eval = eval_ds.map(
        tokenize_fn, batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenize val",
    ) if eval_ds else None

    # نمایش چند نمونه
    for i in range(min(2, len(tokenized_train))):
        ex = tokenized_train[i]
        labels_real = [l for l in ex["labels"] if l != -100]
        inp_dec = tokenizer.decode(ex["input_ids"][:50], skip_special_tokens=True)
        lab_dec = tokenizer.decode(labels_real[:20], skip_special_tokens=True)
        print(f"\nنمونه {i+1}:")
        print(f"  input ({len(ex['input_ids'])} tokens): {inp_dec[:60]}...")
        print(f"  label ({len(labels_real)} tokens): {lab_dec[:60]}...")

    # ── LoRA ────────────────────────────────────────────────
    print("\nاضافه کردن LoRA adapters ...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"قابل آموزش: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if trainable == 0:
        raise RuntimeError("هیچ پارامتر قابل آموزشی وجود ندارد!")

    # ── Data Collator ────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Training Args ────────────────────────────────────────
    # تشخیص precision
    use_fp16 = DEVICE == "cuda" and not torch.cuda.is_bf16_supported()
    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

    if smoke_test:
        train_args = Seq2SeqTrainingArguments(
            output_dir=str(OUTPUT_DIR / "smoke_test"),
            max_steps=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            eval_strategy="steps",
            eval_steps=3,
            save_strategy="steps",
            save_steps=3,
            predict_with_generate=True,
            generation_max_length=64,
            generation_num_beams=2,
            seed=SEED,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=1,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
    else:
        train_args = Seq2SeqTrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=True,
            seed=SEED,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=LOG_STEPS,
            report_to="none",
            predict_with_generate=True,
            generation_max_length=MAX_TARGET_LENGTH,
            generation_num_beams=4,
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )

    # ── Trainer ──────────────────────────────────────────────
    compute_metrics = build_compute_metrics(tokenizer)

    callbacks = []
    if not smoke_test:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── اجرای آموزش ─────────────────────────────────────────
    print(f"\n{'Smoke Test' if smoke_test else 'آموزش کامل'} شروع شد ...")
    t0 = time.time()
    result = trainer.train()
    duration = time.time() - t0
    print(f"\n✓ آموزش کامل شد در {duration/60:.1f} دقیقه")
    print(f"  Training loss: {result.training_loss:.4f}")

    # ── ذخیره مدل ───────────────────────────────────────────
    if not smoke_test:
        print("\nذخیره مدل ...")

        # LoRA adapter
        adapter_dir = OUTPUT_DIR / "lora_adapter"
        adapter_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"✓ LoRA adapter: {adapter_dir}")

        # مدل merge‌شده
        try:
            print("Merge کردن LoRA با مدل پایه ...")
            merged = model.merge_and_unload()
            merged_dir = OUTPUT_DIR / "merged"
            merged_dir.mkdir(exist_ok=True)
            merged.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"✓ مدل merge‌شده: {merged_dir}")
        except Exception as e:
            print(f"⚠️  merge ناموفق: {e}")

        # metadata
        metadata = {
            "training_time_minutes": round(duration / 60, 1),
            "training_loss": result.training_loss,
            "device": DEVICE,
            "model_id": MODEL_ID,
            "seed": SEED,
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
            },
            "training": {
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "effective_batch": EFFECTIVE_BATCH,
                "lr": LEARNING_RATE,
                "fp16": use_fp16,
                "bf16": use_bf16,
            },
        }

        with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n✓ همه فایل‌ها ذخیره شدند")
        print(f"  → {OUTPUT_DIR}")

    return result


# ══ CELL 8: اجرای اصلی ═══════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_training(smoke_test=args.smoke_test)