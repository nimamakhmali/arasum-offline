
## بخش دوم: تسک‌های کامل پروژه

---

### فاز صفر: راه‌اندازی پروژه

---

#### **TASK-001: ایجاد مخزن گیت‌هاب و ساختار اولیه پروژه**

**هدف:** راه‌اندازی پایه پروژه روی گیت‌هاب و ایجاد تمام پوشه‌ها و فایل‌های اسکلتی.

**توضیحات:**
مخزن را با نام انتخاب‌شده روی گیت‌هاب بسازید. تمام پوشه‌های تعریف‌شده در معماری را ایجاد کنید. برای پوشه‌های خالی فایل `.gitkeep` بگذارید. فایل `README.md` اولیه را با عنوان پروژه، توضیح کوتاه، وضعیت (In Development) و بخش‌های اصلی مستند کنید. لایسنس را اضافه کنید. فایل `.gitignore` مناسب برای پایتون بنویسید که شامل موارد زیر باشد: پوشه‌های `models/`، `data/raw/`، `data/processed/`، `storage/`، فایل‌های `.env`، `__pycache__`، `.onnx`، `.gguf` و محیط‌های مجازی.

**خروجی:** مخزن آماده روی گیت‌هاب با ساختار کامل پوشه‌بندی.

---

#### **TASK-002: تعریف و راه‌اندازی محیط توسعه**

**هدف:** ایجاد محیط پایتون ایزوله و یکپارچه برای کل تیم.

**توضیحات:**
فایل `pyproject.toml` را با اطلاعات پروژه (نام، نسخه، توضیح، نویسنده) و تعریف پکیج `arabic_summarizer` از مسیر `src` تنظیم کنید. نسخه پایتون را `3.10+` مشخص کنید. فایل `requirements.txt` را به سه بخش تقسیم کنید:
- `requirements.txt` برای وابستگی‌های اصلی production
- `requirements-dev.txt` برای ابزارهای توسعه مثل pytest، black، flake8
- `requirements-train.txt` برای وابستگی‌های مربوط به Fine-tuning که فقط روی سرور آموزش نیاز است

فایل `.env.example` را با تمام متغیرهای محیطی مورد نیاز بدون مقدار واقعی بسازید. یک `Makefile` ساده با دستورات `install`، `test`، `lint`، `run-api`، `run-ui` بسازید تا اجرای دستورات راحت‌تر شود.

**خروجی:** محیط توسعه قابل راه‌اندازی یکسان روی هر سیستم.

---

#### **TASK-003: پیاده‌سازی سیستم لاگ‌گیری و بارگذاری تنظیمات**

**هدف:** ایجاد زیرساخت مشترک برای لاگ‌گیری و مدیریت تنظیمات که همه ماژول‌ها از آن استفاده می‌کنند.

**توضیحات:**
فایل `src/arabic_summarizer/utils/logger.py` را پیاده‌سازی کنید. از کتابخانه استاندارد `logging` پایتون استفاده کنید. لاگ‌ها باید همزمان در console و فایل نوشته شوند. فرمت لاگ شامل timestamp، سطح، نام ماژول و پیام باشد. سطح لاگ باید از متغیر محیطی خوانده شود.

فایل `src/arabic_summarizer/utils/config_loader.py` را پیاده‌سازی کنید. این کلاس باید فایل‌های YAML را از پوشه `configs` بخواند. باید singleton pattern داشته باشد تا فایل‌ها فقط یک‌بار خوانده شوند. باید مقادیر پیش‌فرض داشته باشد که اگر فایل وجود نداشت سیستم crash نکند.

فایل‌های YAML تنظیمات را با مقادیر واقعی پر کنید:
- `model_config.yaml`: مسیر مدل، حداکثر و حداقل طول خروجی، نوع engine
- `api_config.yaml`: host، port، حداکثر حجم متن ورودی، timeout
- `cache_config.yaml`: فعال/غیرفعال بودن کش، مسیر ذخیره، مدت انقضا

**خروجی:** دو فایل utility کاملاً کارکردی و تنظیمات پایه.

---

#### **TASK-004: تعریف exceptions سراسری پروژه**

**هدف:** ایجاد سلسله‌مراتب exception های سفارشی برای مدیریت خطاهای معنادار.

**توضیحات:**
فایل `src/arabic_summarizer/exceptions.py` را بسازید. یک کلاس پایه `ArabicSummarizerError` تعریف کنید که از `Exception` ارث می‌برد. از آن exception های زیر را مشتق کنید:
- `InvalidInputError`: وقتی متن ورودی خارج از محدوده ۳۰۰ تا ۳۰۰۰ کلمه است
- `ModelNotFoundError`: وقتی فایل مدل در مسیر مشخص‌شده وجود ندارد
- `InferenceError`: وقتی در حین تولید خلاصه خطا رخ می‌دهد
- `PreprocessingError`: وقتی پیش‌پردازش متن شکست می‌خورد
- `UnsupportedLanguageError`: وقتی متن ورودی عربی نیست

هر exception باید یک پیام پیش‌فرض مناسب داشته باشد.

**خروجی:** فایل exceptions کامل.

---

### فاز یک: تحلیل الزامات و آماده‌سازی داده

---

#### **TASK-101: نوشتن سند نیازمندی‌ها و معماری سیستم**

**هدف:** مستندسازی کامل الزامات فنی و غیرفنی پروژه به عنوان مرجع تیم.

**توضیحات:**
فایل `docs/architecture.md` را بنویسید. این سند باید شامل بخش‌های زیر باشد:

**بخش ۱ - نمای کلی سیستم:** شرح معماری چهار لایه‌ای (Preprocessing، Model، Inference، API/UI) به همراه دیاگرام جریان داده به صورت ASCII یا Mermaid.

**بخش ۲ - الزامات عملکردی:** دقیقاً همان‌هایی که پروپوزال مشخص کرده:
- ورودی: ۳۰۰ تا ۳۰۰۰ کلمه عربی
- خروجی: ۱۰٪ تا ۳۰٪ حجم ورودی
- زمان پاسخ قبل از بهینه‌سازی: زیر ۶۰ ثانیه
- زمان پاسخ بعد از بهینه‌سازی: زیر ۱۵ ثانیه
- معیارهای ROUGE و BERTScore

**بخش ۳ - الزامات غیرعملکردی:** آفلاین بودن کامل، پشتیبانی از CPU و GPU، مصرف RAM قابل قبول، قابلیت نصب با Docker.

**بخش ۴ - تصمیمات معماری:** چرا AraBART؟ چرا ONNX؟ چرا FastAPI؟ هر تصمیم باید توجیه داشته باشد.

**بخش ۵ - محدودیت‌ها:** چه چیزهایی خارج از scope هستند.

**خروجی:** فایل `docs/architecture.md` کامل.

---

#### **TASK-102: بررسی و مقایسه مدل‌های پایه عربی**

**هدف:** انتخاب مستند و مستدل بهترین مدل پایه برای Fine-tuning.

**توضیحات:**
فایل `docs/model_selection_report.md` را بنویسید و اسکریپت `scripts/evaluation/compare_base_models.py` را آماده کنید.

مدل‌های کاندیدا را بررسی کنید:
- **AraBART** (moussaKam/AraBART): معماری Encoder-Decoder، pretrain شده برای summarization عربی
- **AraT5** (UBC-NLP/AraT5): مبتنی بر T5، چندوظیفه‌ای
- **mBART-50**: چندزبانه، پشتیبانی از عربی
- **mt5-base**: نسخه چندزبانه T5

برای هر مدل موارد زیر را مقایسه کنید:
- تعداد پارامترها و حجم فایل
- نتایج ROUGE گزارش‌شده در مقالات اصلی
- سرعت inference روی CPU
- کیفیت tokenizer برای عربی
- سهولت fine-tuning با LoRA

در انتها یک مدل را به عنوان **مدل اصلی** و یک مدل را به عنوان **جایگزین پشتیبان** انتخاب و توجیه کنید.

**خروجی:** سند مقایسه مدل‌ها و تصمیم نهایی مستند.

---

#### **TASK-103: جمع‌آوری و بررسی دیتاست‌های استاندارد**

**هدف:** دانلود و بررسی اولیه دیتاست‌های استاندارد خلاصه‌سازی عربی.

**توضیحات:**
اسکریپت `scripts/data/download_datasets.py` را بسازید.

**دیتاست EASC (Egyptian Arabic Summarization Corpus):**
این دیتاست شامل ۱۵۳ مقاله با خلاصه‌های انسانی است. آن را از HuggingFace یا منابع دانشگاهی دانلود کنید. ساختار آن را بررسی کنید: ستون‌های متن اصلی و خلاصه مرجع.

**دیتاست‌های HuggingFace:**
- `csebuetnlp/xlsum` بخش عربی
- `abdalrahmanshahrour/auto-arabic-summarization`
- `Yiyan/arabic_news_summarization`

برای هر دیتاست موارد زیر را مستند کنید:
- تعداد نمونه‌های train/validation/test
- میانگین طول متن ورودی و خروجی
- نوع محتوا (خبری، علمی، دینی)
- کیفیت کلی با بررسی ۵۰ نمونه تصادفی

نمونه‌های خارج از محدوده ۳۰۰ تا ۳۰۰۰ کلمه را شناسایی کنید تا در مرحله بعد فیلتر شوند.

فایل‌های raw را در `data/raw/` ذخیره کنید.

**خروجی:** دیتاست‌های دانلود شده و گزارش بررسی اولیه.

---

#### **TASK-104: پیاده‌سازی ماژول نرمال‌سازی متون عربی**

**هدف:** پیاده‌سازی ماژول تخصصی نرمال‌سازی حروف و متن عربی.

**توضیحات:**
فایل `src/arabic_summarizer/preprocessing/normalizer.py` را پیاده‌سازی کنید.

کلاس `ArabicNormalizer` باید متدهای زیر را داشته باشد:

**normalize_alef:** تبدیل تمام اشکال الف (أ، إ، آ، ٱ) به شکل ساده (ا).

**normalize_teh_marbuta:** یکسان‌سازی تاء مربوطه.

**normalize_yeh:** یکسان‌سازی ی و ى.

**remove_diacritics:** حذف اعراب (حرکات) شامل فتحه، کسره، ضمه، سکون، شدّه، تنوین و مدّ. این کار با Unicode range `\u064B` تا `\u065F` انجام می‌شود.

**remove_tatweel:** حذف کشیده (ـ) که برای زیبایی متن استفاده می‌شود.

**normalize_whitespace:** حذف فاصله‌های اضافه و یکسان‌سازی فاصله‌ها.

**متد اصلی normalize:** همه متدهای بالا را به ترتیب اجرا می‌کند.

تمام عملیات با Regular Expression انجام می‌شود. کتابخانه `re` پایتون کافی است. هیچ وابستگی خارجی نداشته باشد.

**خروجی:** فایل `normalizer.py` با unit test مربوطه.

---

#### **TASK-105: پیاده‌سازی ماژول پاک‌سازی متون عربی**

**هدف:** پیاده‌سازی ماژول حذف نویز، نشانه‌گذاری و محتوای غیرضروری.

**توضیحات:**
فایل `src/arabic_summarizer/preprocessing/cleaner.py` را پیاده‌سازی کنید.

کلاس `ArabicTextCleaner` باید متدهای زیر را داشته باشد:

**remove_urls:** حذف آدرس‌های اینترنتی با regex استاندارد.

**remove_html_tags:** حذف تگ‌های HTML برای متونی که از وب scrappe شده‌اند.

**remove_non_arabic:** حذف کاراکترهای غیرعربی و غیرضروری. توجه: اعداد، نقطه، ویرگول و علامت سوال عربی باید حفظ شوند.

**remove_repeated_punctuation:** تبدیل `!!!` به `!` و `...` به `.`.

**remove_extra_newlines:** تبدیل چند خط خالی پشت سرهم به یک خط.

**clean_for_summarization:** متد اصلی که تمام مراحل را به ترتیب صحیح اجرا می‌کند.

**validate_input_length:** تعداد کلمات متن را می‌شمارد و اگر خارج از محدوده ۳۰۰ تا ۳۰۰۰ بود `InvalidInputError` raise می‌کند.

تعداد کلمات عربی با split روی whitespace و فیلتر توکن‌های خالی محاسبه می‌شود.

**خروجی:** فایل `cleaner.py` با unit test مربوطه.

---

#### **TASK-106: ادغام و آزمون خط‌لوله پیش‌پردازش با CamelTools**

**هدف:** یکپارچه‌سازی ماژول‌های normalizer و cleaner با CamelTools و ساخت pipeline نهایی.

**توضیحات:**
کتابخانه `camel-tools` را نصب و بررسی کنید. این کتابخانه ابزارهای تخصصی پردازش متن عربی دارد. بخش‌هایی که برای این پروژه مفید هستند:
- `camel_tools.utils.normalize`: normalize کردن حروف عربی
- `camel_tools.tokenizers.word`: توکن‌سازی کلمات

در فایل `src/arabic_summarizer/preprocessing/__init__.py` کلاس `ArabicPreprocessingPipeline` را بسازید که:
- ابتدا `ArabicTextCleaner` را اجرا می‌کند
- سپس `ArabicNormalizer` را اجرا می‌کند
- در صورت موجود بودن CamelTools، از normalize آن هم استفاده می‌کند
- اگر CamelTools نصب نبود، gracefully به ماژول‌های داخلی fall back می‌کند

یک اسکریپت تست سریع بنویسید که ۱۰ متن عربی نمونه را از `resources/sample_texts/` بخواند و نتیجه pipeline را چاپ کند. نمونه‌های متنی از انواع مختلف (خبری، دینی، رسمی) در `resources/sample_texts/` قرار دهید.

**خروجی:** Pipeline پیش‌پردازش یکپارچه و تست‌شده.

---

#### **TASK-107: پیاده‌سازی اسکریپت آماده‌سازی دیتاست برای آموزش**

**هدف:** تبدیل دیتاست‌های خام به فرمت استاندارد مورد نیاز Fine-tuning.

**توضیحات:**
اسکریپت `scripts/data/prepare_dataset.py` را بسازید.

این اسکریپت باید:

**مرحله ۱ - بارگذاری:** دیتاست‌های raw را از `data/raw/` بخواند. فرمت‌های CSV، JSON و HuggingFace datasets را پشتیبانی کند.

**مرحله ۲ - فیلترینگ:** نمونه‌هایی که متن اصلی‌شان خارج از ۳۰۰ تا ۳۰۰۰ کلمه است حذف شوند. نمونه‌هایی که خلاصه مرجع ندارند حذف شوند. نمونه‌های تکراری با بررسی hash متن حذف شوند.

**مرحله ۳ - پیش‌پردازش:** روی هر نمونه Pipeline پیش‌پردازش فاز ۱۰۶ را اجرا کند.

**مرحله ۴ - تقسیم‌بندی:** دیتاست را به نسبت ۸۰/۱۰/۱۰ برای train/validation/test تقسیم کند. تقسیم‌بندی باید random seed ثابت (مثلاً ۴۲) داشته باشد.

**مرحله ۵ - ذخیره:** در فرمت `jsonl` (هر خط یک JSON) در `data/processed/` ذخیره کند. یک فایل `data/processed/dataset_stats.json` بسازد که شامل تعداد نمونه‌های هر split، میانگین طول و توزیع طول متون باشد.

**خروجی:** دیتاست آماده برای آموزش در `data/processed/`.

---

#### **TASK-108: نوشتن unit test های فاز پیش‌پردازش**

**هدف:** اطمینان از صحت عملکرد تمام ماژول‌های پیش‌پردازش با تست‌های اتوماتیک.

**توضیحات:**
فایل `tests/unit/test_preprocessing.py` را بسازید.

تست‌های `ArabicNormalizer`:
- ورودی `أحمد` باید به `احمد` تبدیل شود
- ورودی `مَدْرَسَة` باید اعراب نداشته باشد
- ورودی `إبراهيم` باید `ابراهيم` شود
- متن بدون نیاز به نرمال‌سازی نباید تغییر کند

تست‌های `ArabicTextCleaner`:
- URL باید حذف شود
- تگ HTML باید حذف شود
- متن ۲۹۹ کلمه باید `InvalidInputError` بدهد
- متن ۳۰۰۱ کلمه باید `InvalidInputError` بدهد
- متن ۵۰۰ کلمه باید بدون خطا پردازش شود

تست‌های Pipeline:
- ترتیب اجرای مراحل صحیح باشد
- خروجی pipeline برای یک متن نمونه واقعی عربی بررسی شود

از `pytest` و `pytest-parametrize` برای تست‌های parametric استفاده کنید.

**خروجی:** مجموعه تست با coverage بالای ۸۰٪ برای ماژول‌های preprocessing.

---

### فاز دو: Fine-Tuning مدل

---

#### **TASK-201: راه‌اندازی محیط آموزش ابری**

**هدف:** آماده‌سازی محیط Google Colab Pro یا سرویس ابری برای Fine-tuning.

**توضیحات:**
یک Jupyter Notebook به نام `finetune_setup.ipynb` در پوشه `scripts/training/` بسازید که مراحل زیر را step-by-step داشته باشد:

- نصب وابستگی‌های آموزش: `transformers`، `datasets`، `peft`، `accelerate`، `bitsandbytes`
- اتصال به Google Drive برای ذخیره checkpoint ها
- بررسی GPU موجود و حافظه آن
- دانلود مدل پایه انتخاب‌شده از HuggingFace
- بارگذاری دیتاست processed از `data/processed/`
- یک اجرای آزمایشی با ۱۰۰ نمونه برای اطمینان از صحت pipeline

فایل `requirements-train.txt` را با تمام وابستگی‌های لازم برای این مرحله کامل کنید. نسخه‌های دقیق را pin کنید تا reproducible باشد.

**خروجی:** محیط آموزش کاملاً آماده و تست‌شده.

---

#### **TASK-202: پیاده‌سازی اسکریپت Fine-Tuning با LoRA/QLoRA**

**هدف:** Fine-tuning مدل پایه عربی روی دیتاست خلاصه‌سازی با کمترین منابع GPU.

**توضیحات:**
فایل `scripts/training/finetune.py` را بسازید.

تنظیمات LoRA: از `peft` library استفاده کنید. `r=16`، `lora_alpha=32`، `target_modules` را بر اساس معماری مدل انتخابی تنظیم کنید. `dropout=0.1`.

تنظیمات آموزش:
- `learning_rate=5e-5` (شروع)
- `num_train_epochs=3`
- `batch_size=4` (یا کمتر بر اساس GPU)
- `gradient_accumulation_steps=4`
- `evaluation_strategy="epoch"`
- `save_strategy="epoch"` با نگه داشتن بهترین checkpoint

Callback های ضروری:
- Early stopping با `patience=2`
- لاگ‌گیری loss به فایل CSV برای رسم نمودار

در پایان هر epoch، معیارهای ROUGE را روی validation set محاسبه و لاگ کنید.

مدل نهایی را در `models/finetuned/` ذخیره کنید.

**خروجی:** مدل Fine-tuned ذخیره‌شده با گزارش training loss و ROUGE.

---

#### **TASK-203: ارزیابی مدل Fine-tuned**

**هدف:** سنجش دقیق کیفیت مدل با معیارهای تعریف‌شده در پروپوزال.

**توضیحات:**
اسکریپت `scripts/evaluation/run_rouge_eval.py` را بسازید.

مدل Fine-tuned را روی test set اجرا کنید. برای هر نمونه خلاصه تولید کنید و نگه دارید.

محاسبه ROUGE: از کتابخانه `rouge-score` استفاده کنید. ROUGE-1، ROUGE-2 و ROUGE-L را برای هر نمونه و به صورت میانگین کل محاسبه کنید. هدف: ROUGE-1 ≥ 0.45، ROUGE-2 ≥ 0.25، ROUGE-L ≥ 0.40.

محاسبه BERTScore: از کتابخانه `bert-score` استفاده کنید. مدل مناسب برای عربی را انتخاب کنید (مثلاً `bert-base-multilingual-cased`). F1 Score را محاسبه کنید. هدف: BERTScore F1 ≥ 0.85.

اندازه‌گیری Latency: زمان تولید خلاصه برای ۵۰ نمونه با طول‌های مختلف اندازه بگیرید. مطمئن شوید که روی سخت‌افزار هدف (بدون GPU) زیر ۶۰ ثانیه است.

نتایج را در `data/eval/finetuned_evaluation.json` ذخیره کنید.

اگر هر معیار به هدف نرسید، دلایل را تحلیل و راهکار تنظیم hyperparameter پیشنهاد دهید.

**خروجی:** گزارش ارزیابی کامل مدل Fine-tuned.

---

### فاز سه: کوانتیزه‌سازی و بهینه‌سازی

---

#### **TASK-301: تبدیل مدل به فرمت ONNX**

**هدف:** تبدیل مدل Fine-tuned به فرمت ONNX برای inference سریع روی CPU.

**توضیحات:**
اسکریپت `scripts/quantization/export_and_quantize.py` را بسازید.

**مرحله Export:**
از `optimum` کتابخانه HuggingFace برای export استفاده کنید. مدل Encoder و Decoder را جداگانه export کنید (برای Seq2Seq این الزامی است). فایل‌های ONNX را در `models/quantized/onnx_fp32/` ذخیره کنید.

**مرحله Quantization:**
کوانتیزه‌سازی INT8 با `onnxruntime.quantization` انجام دهید. Dynamic Quantization برای Linear layers اجرا کنید. فایل‌های کوانتیزه را در `models/quantized/onnx_int8/` ذخیره کنید.

**مرحله تست اولیه:**
با یک متن نمونه خروجی مدل ONNX را با مدل اصلی مقایسه کنید تا مطمئن شوید صحت حفظ شده.

**خروجی:** فایل‌های ONNX اصلی و کوانتیزه در `models/quantized/`.

---

#### **TASK-302: پیاده‌سازی موتور inference با ONNX Runtime**

**هدف:** پیاده‌سازی کلاس inference بهینه با ONNX Runtime برای اجرای آفلاین.

**توضیحات:**
فایل `src/arabic_summarizer/inference/base_engine.py` را بسازید. یک Abstract Base Class به نام `BaseInferenceEngine` تعریف کنید با متدهای abstract:
- `load_model(model_path)`: بارگذاری مدل
- `generate_summary(text, min_length, max_length)`: تولید خلاصه
- `get_model_info()`: اطلاعات مدل لود شده

فایل `src/arabic_summarizer/inference/onnx_engine.py` را پیاده‌سازی کنید:
- `OnnxInferenceEngine` که از `BaseInferenceEngine` ارث می‌برد
- بارگذاری session با `onnxruntime.InferenceSession`
- تنظیم `SessionOptions` برای استفاده بهینه از thread های CPU
- پیاده‌سازی beam search برای تولید متن اگر ONNX Runtime آن را support نکرد
- مدیریت حافظه: آزاد کردن session وقتی لازم نیست
- logging زمان inference برای هر درخواست

**خروجی:** موتور inference با ONNX Runtime کاملاً کارکردی.

---

#### **TASK-303: بنچ‌مارک سرعت و مقایسه عملکرد**

**هدف:** مستندسازی دقیق عملکرد مدل قبل و بعد از بهینه‌سازی.

**توضیحات:**
اسکریپت `scripts/evaluation/benchmark_speed.py` را بسازید.

بنچ‌مارک باید سه حالت را مقایسه کند:
1. مدل Fine-tuned اصلی (PyTorch)
2. مدل ONNX FP32
3. مدل ONNX INT8

برای هر حالت و هر طول ورودی (۳۰۰، ۶۰۰، ۱۰۰۰، ۲۰۰۰، ۳۰۰۰ کلمه) اندازه بگیرید:
- زمان اولین inference (cold start)
- میانگین زمان inference در ۱۰ اجرا (warm)
- حداکثر مصرف RAM حین inference
- کیفیت ROUGE روی همان نمونه‌های مرجع

نتایج را در یک جدول markdown ذخیره کنید و در `data/eval/benchmark_results.json` هم ذخیره کنید.

هدف: مدل ONNX INT8 باید روی Intel Core i7-12700H زیر ۱۵ ثانیه برای متن ۱۰۰۰ کلمه‌ای باشد.

**خروجی:** گزارش بنچ‌مارک کامل برای گزارش فنی نهایی.

---

### فاز چهار: پیاده‌سازی API و کتابخانه

---

#### **TASK-401: پیاده‌سازی ماژول‌های postprocessing و hashing**

**هدف:** تکمیل ماژول‌های کمکی باقیمانده.

**توضیحات:**
فایل `src/arabic_summarizer/postprocessing/formatter.py` را بسازید. کلاس `SummaryFormatter` که:
- خلاصه تولید شده را تمیز می‌کند (حذف توکن‌های خاص مثل `</s>`, `[PAD]`)
- مطمئن می‌شود متن با حرف بزرگ شروع می‌شود (در عربی یعنی بدون فاصله اضافه ابتدا)
- طول خروجی را در محدوده درخواستی نگه می‌دارد

فایل `src/arabic_summarizer/utils/hashing.py` را بسازید:
- تابعی که یک متن را می‌گیرد و hash SHA-256 آن را برمی‌گرداند
- این hash برای کش کردن نتایج استفاده می‌شود
- ورودی‌های یکسان با تنظیمات یکسان باید hash یکسان تولید کنند

**خروجی:** دو فایل utility کامل.

---

#### **TASK-402: پیاده‌سازی سیستم Cache**

**هدف:** ایجاد سیستم کش برای جلوگیری از پردازش مجدد درخواست‌های تکراری.

**توضیحات:**
فایل `src/arabic_summarizer/cache/cache_manager.py` را پیاده‌سازی کنید.

کلاس `CacheManager` که:
- نتایج inference را در فایل‌های JSON در `storage/cache/` ذخیره می‌کند
- key کش: hash متن ورودی + درصد خلاصه‌سازی است
- قبل از inference بررسی می‌کند آیا این درخواست قبلاً پردازش شده
- اگر cache hit بود نتیجه ذخیره‌شده را برمی‌گرداند
- انقضای کش بر اساس تنظیمات `cache_config.yaml`
- قابلیت clear کردن کل کش یا یک entry خاص

کش باید thread-safe باشد (برای API server چند thread‌ای).

**خروجی:** سیستم cache کارکردی.

---

#### **TASK-403: پیاده‌سازی Pipeline اصلی و کلاس Summarizer**

**هدف:** یکپارچه‌سازی تمام ماژول‌ها در یک interface ساده و منسجم.

**توضیحات:**
فایل `src/arabic_summarizer/pipeline.py` را بسازید. کلاس `SummarizationPipeline` که:
- در `__init__` تنظیمات را از config می‌خواند و موتور inference را load می‌کند
- متد `process(text, summary_ratio)` که:
  1. ورودی را validate می‌کند
  2. کش را بررسی می‌کند
  3. پیش‌پردازش را اجرا می‌کند
  4. inference را اجرا می‌کند
  5. postprocessing را اجرا می‌کند
  6. نتیجه را کش می‌کند
  7. خلاصه، زمان inference و تعداد کلمات را برمی‌گرداند

فایل `src/arabic_summarizer/summarizer.py` را بسازید. این فایل interface عمومی کتابخانه است. یک تابع ساده `summarize(text, ratio=0.2)` که Pipeline را مدیریت می‌کند. این همان چیزی است که کاربر نهایی با `from arabic_summarizer import summarize` استفاده می‌کند.

**خروجی:** API عمومی کتابخانه پایتون.

---

#### **TASK-404: پیاده‌سازی Schema های API**

**هدف:** تعریف مدل‌های داده ورودی و خروجی API با Pydantic.

**توضیحات:**
فایل `api/schemas/request_models.py` را بسازید:
- `SummarizeRequest`: شامل `text: str`، `summary_ratio: float` (بین ۰.۱ تا ۰.۳ با مقدار پیش‌فرض ۰.۲)
- validation با Pydantic: طول متن، محدوده ratio
- field descriptions برای مستندات Swagger

فایل `api/schemas/response_models.py` را بسازید:
- `SummarizeResponse`: شامل `summary: str`، `word_count_original: int`، `word_count_summary: int`، `compression_ratio: float`، `inference_time_seconds: float`، `cached: bool`
- `HealthResponse`: شامل `status: str`، `model_loaded: bool`، `model_name: str`

**خروجی:** Schema های کامل با validation.

---

#### **TASK-405: پیاده‌سازی Router های API**

**هدف:** پیاده‌سازی endpoint های FastAPI.

**توضیحات:**
فایل `api/routers/health.py` را بسازید:
- `GET /health`: وضعیت سرویس و مدل را برمی‌گرداند
- اگر مدل load نشده باشد status را `degraded` برمی‌گرداند

فایل `api/routers/summarize.py` را بسازید:
- `POST /api/v1/summarize`: متن عربی دریافت کرده و خلاصه برمی‌گرداند
- validation ورودی با Pydantic
- مدیریت خطاها: `InvalidInputError` با HTTP 422، `InferenceError` با HTTP 500

فایل `api/dependencies.py` را بسازید:
- تابع dependency که instance آماده Pipeline را برمی‌گرداند (Dependency Injection)
- Pipeline باید یک‌بار load شود نه برای هر request

فایل `api/exception_handlers.py` را بسازید:
- handler برای `ArabicSummarizerError` که response JSON مناسب برمی‌گرداند

فایل `api/main.py` را بسازید:
- ایجاد app FastAPI با عنوان، نسخه و توضیح
- ثبت router ها و exception handler ها
- middleware برای logging هر request

**خروجی:** API کاملاً کارکردی.

---

#### **TASK-406: نوشتن تست‌های API و Pipeline**

**هدف:** اطمینان از صحت عملکرد API و Pipeline با تست‌های integration.

**توضیحات:**
فایل `tests/integration/test_api_endpoints.py` را بسازید. از `httpx` و `pytest-asyncio` استفاده کنید. از `TestClient` فاست‌ای استفاده کنید تا نیازی به مدل واقعی نباشد (mock کنید).

تست‌های ضروری:
- `GET /health` باید ۲۰۰ و JSON معتبر برگرداند
- `POST /api/v1/summarize` با متن معتبر باید ۲۰۰ برگرداند
- `POST /api/v1/summarize` با متن کوتاه‌تر از ۳۰۰ کلمه باید ۴۲۲ برگرداند
- `POST /api/v1/summarize` با ratio خارج از محدوده باید ۴۲۲ برگرداند
- بررسی ساختار response

فایل `tests/unit/test_pipeline.py` را بسازید که Pipeline را با mock کردن inference engine تست کند.

**خروجی:** مجموعه تست integration کامل.

---

### فاز پنج: رابط کاربری و بسته‌بندی

---

#### **TASK-501: پیاده‌سازی رابط کاربری Gradio**

**هدف:** ایجاد رابط کاربری تعاملی تحت وب برای تست و نمایش سیستم.

**توضیحات:**
فایل `ui/app_gradio.py` را پیاده‌سازی کنید.

رابط کاربری باید شامل باشد:

**بخش ورودی:**
- یک textbox بزرگ برای وارد کردن متن عربی
- یک slider برای تنظیم درصد خلاصه‌سازی (۱۰٪ تا ۳۰٪)
- دکمه "خلاصه‌سازی"

**بخش خروجی:**
- نمایش خلاصه تولید شده
- نمایش آماری: تعداد کلمات اصلی، تعداد کلمات خلاصه، نسبت فشرده‌سازی، زمان پردازش
- نشانگر cache hit/miss

**تنظیمات UI:**
- راستچین برای متون عربی (با CSS سفارشی)
- یک tab برای وارد کردن متن دستی
- یک tab برای نمونه‌های از پیش تعریف‌شده (۵ متن نمونه از `resources/sample_texts/`)
- عنوان و توضیح پروژه

اجرا باید کاملاً آفلاین باشد: `launch(server_name="127.0.0.1", share=False)`.

**خروجی:** رابط کاربری کاملاً کارکردی آفلاین.

---

#### **TASK-502: تهیه Dockerfile و Docker Compose**

**هدف:** بسته‌بندی کل پروژه در Docker برای استقرار آسان.

**توضیحات:**
فایل `Dockerfile` را بسازید:
- از `python:3.10-slim` base image استفاده کنید
- وابستگی‌های سیستم لازم برای ONNX Runtime را نصب کنید
- requirements را نصب کنید
- پکیج `arabic_summarizer` را با pip install -e نصب کنید
- فایل‌های مدل باید با volume mount در runtime ارائه شوند (نه داخل image)
- `EXPOSE 8000` برای API
- CMD برای اجرای FastAPI با uvicorn

فایل `docker-compose.yml` را بسازید:
- service `api`: اجرای FastAPI روی پورت ۸۰۰۰
- service `ui`: اجرای Gradio روی پورت ۷۸۶۰
- volume mount برای پوشه `models/` و `storage/`
- environment variables از `.env`

راهنمای نصب و اجرا را در `docs/installation_guide.md` بنویسید که شامل:
- نصب بدون Docker (محیط مجازی)
- نصب با Docker
- دانلود مدل و قرار دادن در پوشه صحیح
- اجرای اولیه و تست

**خروجی:** Docker setup کاملاً کارکردی و راهنمای نصب.

---

#### **TASK-503: ایجاد پکیج پایتون قابل نصب**

**هدف:** تبدیل کتابخانه به یک پکیج استاندارد پایتون قابل نصب با pip.

**توضیحات:**
فایل `pyproject.toml` را کامل کنید با:
- اطلاعات پروژه (name, version, description, authors)
- وابستگی‌های اجباری با نسخه‌های دقیق
- optional dependencies: `[training]` برای Fine-tuning، `[ui]` برای Gradio
- entry points: یک CLI ساده `arabic-summarizer` که می‌توان از terminal استفاده کرد

یک CLI ساده در `src/arabic_summarizer/__main__.py` بسازید که:
- متن را از stdin یا argument می‌خواند
- خلاصه را به stdout چاپ می‌کند
- پرچم `--ratio` برای تنظیم نسبت خلاصه

**خروجی:** پکیج pip-installable کامل.

---

### فاز شش: ارزیابی نهایی و مستندسازی

---

#### **TASK-601: ارزیابی نهایی کامل سیستم**

**هدف:** اجرای ارزیابی جامع بر روی سخت‌افزار هدف و تولید گزارش نهایی.

**توضیحات:**
ماژول `src/arabic_summarizer/evaluation/rouge_scorer.py` را بسازید که ROUGE را محاسبه می‌کند و قابل import توسط script های دیگر باشد.

ماژول `src/arabic_summarizer/evaluation/bert_scorer.py` را بسازید که BERTScore را محاسبه می‌کند.

ارزیابی نهایی کامل را روی سخت‌افزار هدف (i7-12700H، ۱۶GB RAM) اجرا کنید. باید تمام معیارها ثبت شوند:
- ROUGE-1، ROUGE-2، ROUGE-L روی test set کامل
- BERTScore F1 روی test set کامل
- Latency برای هر category طول متن
- مصرف RAM حین inference

فایل `docs/evaluation_report.md` را با نتایج کامل، جداول، تحلیل و مقایسه با اهداف پروپوزال بنویسید.

**خروجی:** گزارش ارزیابی نهایی مستند.

---

#### **TASK-602: تکمیل README.md اصلی**

**هدف:** نوشتن README حرفه‌ای که هم برای ارائه به کارفرما و هم برای گیت‌هاب مناسب باشد.

**توضیحات:**
`README.md` باید شامل بخش‌های زیر باشد:
- Badge های وضعیت (Python version, License, Status)
- توضیح کوتاه پروژه به فارسی و عربی
- معیارهای عملکرد رسیده (ROUGE, BERTScore, Latency)
- معماری ساده به صورت دیاگرام
- نصب سریع (Quick Start)
- مثال استفاده از کتابخانه پایتون
- مثال استفاده از API (با curl)
- مثال استفاده از UI
- ساختار پروژه
- محدودیت‌ها
- نقشه راه (پروژه دوم و سوم)
- License

**خروجی:** README حرفه‌ای و جامع.

---

#### **TASK-603: تهیه بسته تحویل نهایی**

**هدف:** آماده‌سازی تمام deliverable های تعریف‌شده در پروپوزال برای تحویل به کارفرما.

**توضیحات:**
چک‌لیست تحویل بر اساس پروپوزال:

- [ ] سورس‌کد کامل روی گیت‌هاب (private repo با دسترسی کارفرما)
- [ ] فایل مدل کوانتیزه ONNX (با لینک دانلود به دلیل حجم)
- [ ] پکیج پایتون (wheel file)
- [ ] Dockerfile و docker-compose
- [ ] `docs/installation_guide.md`: راهنمای نصب و استقرار
- [ ] `docs/evaluation_report.md`: گزارش ارزیابی فنی کامل
- [ ] `docs/architecture.md`: مستندات معماری
- [ ] گزارش بنچ‌مارک CPU
- [ ] یک release روی گیت‌هاب با tag نسخه `v1.0.0`

یک فایل `DELIVERY_CHECKLIST.md` در ریشه پروژه بسازید که وضعیت هر مورد را نشان دهد.

**خروجی:** پروژه آماده تحویل به کارفرما.

---

## خلاصه تسک‌ها

| فاز | تسک | عنوان |
|-----|------|--------|
| ۰ | TASK-001 | راه‌اندازی مخزن و ساختار پروژه |
| ۰ | TASK-002 | محیط توسعه |
| ۰ | TASK-003 | لاگ‌گیری و تنظیمات |
| ۰ | TASK-004 | Exceptions |
| ۱ | TASK-101 | سند نیازمندی‌ها |
| ۱ | TASK-102 | مقایسه مدل‌های پایه |
| ۱ | TASK-103 | جمع‌آوری دیتاست |
| ۱ | TASK-104 | ماژول Normalizer |
| ۱ | TASK-105 | ماژول Cleaner |
| ۱ | TASK-106 | Pipeline پیش‌پردازش |
| ۱ | TASK-107 | اسکریپت آماده‌سازی دیتاست |
| ۱ | TASK-108 | Unit Test های پیش‌پردازش |
| ۲ | TASK-201 | محیط آموزش ابری |
| ۲ | TASK-202 | Fine-Tuning با LoRA |
| ۲ | TASK-203 | ارزیابی مدل Fine-tuned |
| ۳ | TASK-301 | Export به ONNX |
| ۳ | TASK-302 | موتور ONNX Runtime |
| ۳ | TASK-303 | بنچ‌مارک سرعت |
| ۴ | TASK-401 | Postprocessing و Hashing |
| ۴ | TASK-402 | سیستم Cache |
| ۴ | TASK-403 | Pipeline و Summarizer |
| ۴ | TASK-404 | Schema های API |
| ۴ | TASK-405 | Router های API |
| ۴ | TASK-406 | تست‌های API |
| ۵ | TASK-501 | رابط کاربری Gradio |
| ۵ | TASK-502 | Docker |
| ۵ | TASK-503 | پکیج پایتون |
| ۶ | TASK-601 | ارزیابی نهایی |
| ۶ | TASK-602 | README نهایی |
| ۶ | TASK-603 | بسته تحویل |

---
