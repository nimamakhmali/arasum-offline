# راهنمای Fine-Tuning با Google Colab

## گام ۱: باز کردن Colab
→ colab.research.google.com
→ New notebook

## گام ۲: فعال‌سازی GPU
Runtime > Change runtime type > Hardware accelerator: T4 GPU > Save

## گام ۳: نصب وابستگی‌ها (سلول اول)
```python
!pip install -q transformers==4.41.2 peft==0.11.1 accelerate==0.31.0 \
    datasets==2.19.2 rouge-score==0.1.2 bert-score==0.3.13 \
    sentencepiece==0.2.0 protobuf==4.25.3