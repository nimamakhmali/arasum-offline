# گزارش انتخاب مدل پایه

## مدل‌های بررسی‌شده

| مدل | پارامترها | حجم | نوع معماری |
|-----|-----------|-----|------------|
| moussaKam/AraBART | 139M | ~560MB | BART (Enc-Dec) |
| UBC-NLP/AraT5-base | 220M | ~880MB | T5 (Enc-Dec) |
| facebook/mbart-large-cc25 | 610M | ~2.4GB | mBART (Enc-Dec) |
| google/mt5-base | 580M | ~2.3GB | T5 چندزبانه |

## معیارهای مقایسه

### ۱. کیفیت زبان عربی
- AraBART: pretrain شده فقط روی عربی فصیح ✅
- AraT5: چندین variant عربی دارد ✅
- mBART و mT5: چندزبانه، کیفیت عربی متوسط ⚠️

### ۲. سازگاری با Quantization
- مدل‌های کوچک‌تر (AraBART, AraT5-base) بهتر quantize می‌شوند
- مدل‌های بزرگ پس از INT8 quantization حافظه زیادی مصرف می‌کنند

### ۳. سابقه در summarization
- AraBART: fine-tune شده روی EASC در مقاله اصلی، ROUGE-1 ≈ 0.47
- AraT5: نتایج مشابه با fine-tuning مناسب

## تصمیم نهایی

**مدل اصلی: `moussaKam/AraBART`**
دلیل: بهترین نسبت کیفیت به حجم برای عربی فصیح، 
سابقه اثبات‌شده روی EASC، حجم مناسب برای quantization.

**مدل جایگزین: `UBC-NLP/AraT5-base`**
دلیل: در صورتی که AraBART به ROUGE هدف نرسید.