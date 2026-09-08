"""
scripts/training/colab_train.py

اسکریپت یکپارچه برای اجرا در Google Colab.
تمام منطق در یک فایل است تا آپلود راحت باشد.

دستورالعمل Colab:
1. Runtime > Change runtime type > T4 GPU
2. این فایل را آپلود کنید
3. سلول‌ها را به ترتیب اجرا کنید
"""

# ══ سلول ۱: نصب وابستگی‌ها ══════════════════════════════════
INSTALL_CMD = """
pip install -q transformers==4.41.2 peft==0.11.1 accelerate==0.31.0 \\
    datasets==2.19.2 rouge-score==0.1.2 bert-score==0.3.13 \\
    sentencepiece==0.2.0 protobuf==4.25.3
"""

# ══ سلول ۲: بررسی GPU ════════════════════════════════════════
VERIFY_GPU = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    raise RuntimeError("GPU پیدا نشد! Runtime > Change runtime type > GPU")
"""

# ══ سلول ۳: اتصال Google Drive ══════════════════════════════
MOUNT_DRIVE = """
from google.colab import drive
drive.mount('/content/drive')

import os
OUTPUT_DIR = '/content/drive/MyDrive/arasum-offline/models/arabart'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output dir: {OUTPUT_DIR}")
"""

# ══ سلول ۴: آپلود دیتاست ════════════════════════════════════
UPLOAD_DATA = """
# آپلود training_ready.zip از طریق:
# Files panel (سمت چپ) > Upload > training_ready.zip

import zipfile, os
if os.path.exists('/content/training_ready.zip'):
    with zipfile.ZipFile('/content/training_ready.zip', 'r') as z:
        z.extractall('/content/')
    print("دیتاست آپلود و extract شد")
    
DATA_DIR = '/content/data/training_ready/xlsum_arabic'
print(f"Data dir exists: {os.path.exists(DATA_DIR)}")
"""

print("این فایل را در Colab اجرا کنید")
print("دستورالعمل کامل در docs/colab_guide.md است")