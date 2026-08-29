# گزارش انتخاب مدل پایه

## وضعیت فعلی

این گزارش نتیجه بررسی اولیه مدل‌های کاندید است.
**انتخاب نهایی پس از benchmark کنترل‌شده انجام خواهد شد.**

---

## مدل‌های بررسی‌شده

> ⚠️ نکته یکپارچگی: ID های زیر دقیقاً با `configs/model_config.yaml` یکسان است.

| مدل | HuggingFace ID | پارامترها | حجم | معماری |
|-----|---------------|-----------|-----|--------|
| AraBART | `moussaKam/AraBART` | 139M | ~560MB | BART |
| AraT5-base | `UBC-NLP/AraT5-base` | 220M | ~880MB | T5 |
| mT5-small | `google/mt5-small` | 300M | ~1.2GB | T5 |
| mBART-large | `facebook/mbart-large-cc25` | 610M | ~2.4GB | mBART |

---

## معیارهای مقایسه اولیه

### ۱. تخصص زبان عربی

| مدل | نوع | توضیح |
|-----|-----|-------|
| AraBART | تک‌زبانه عربی | pretrain روی عربی فصیح |
| AraT5 | تک‌زبانه عربی | چند variant برای تسک‌های مختلف |
| mT5-small | چندزبانه (101 زبان) | baseline سبک |
| mBART | چندزبانه (50 زبان) | غیرفعال - حجم زیاد |

### ۲. سازگاری با Quantization

- AraBART (~560MB) و AraT5 (~880MB): مناسب برای quantization
- mBART (~2.4GB): پس از INT8 هنوز بزرگ است

### ۳. سابقه در summarization (از مقالات)

- **AraBART**: ROUGE-1 ≈ 0.47 روی EASC (Kamal et al., 2021)
- **AraT5**: نتایج مشابه در generation عربی
- **mT5/mBART**: عملکرد پایین‌تر در عربی فصیح

---

## کاندید اولیه پیشنهادی

**کاندید اصلی اولیه: `moussaKam/AraBART`**

> **جمله دقیق علمی:**
> «AraBART به عنوان کاندید اصلی اولیه انتخاب شد.
> انتخاب نهایی پس از benchmark کنترل‌شده روی حداقل ۲۰ نمونه
> یکسان برای همه مدل‌ها انجام خواهد شد.»

**کاندید جایگزین: `UBC-NLP/AraT5-base`**

---

## ⚠️ محدودیت‌های مهم

### ۱. محدودیت طول ورودی

| مدل | max_input_length | متن 3000 کلمه |
|-----|-----------------|---------------|
| AraBART | 1024 توکن | truncate می‌شود |
| AraT5 | 512 توکن | truncate می‌شود |
| mT5-small | 512 توکن | truncate می‌شود |

> **نتیجه:** پشتیبانی از 3000 کلمه در API ≠ پشتیبانی واقعی مدل.
> در مراحل بعد نیاز به chunking یا hierarchical summarization داریم.

### ۲. Dataset Compression Mismatch

| معیار | Dataset (XL-Sum) | هدف پروژه |
|-------|-----------------|-----------|
| نسبت فشرده‌سازی | ~5% | 10%-30% |

این مغایرت در مرحله Fine-tuning از طریق کنترل `max_new_tokens`
و `min_new_tokens` مدیریت خواهد شد.

---

## برنامه benchmark واقعی (TASK-102)

```bash
python scripts/model_selection/compare_base_models.py \
    --num-samples 20 \
    --device cpu \
    --dataset-path data/processed/xlsum_arabic/validation.jsonl