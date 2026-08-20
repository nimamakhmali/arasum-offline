arabic-offline-summarizer/
│
├── api/                                  # لایه سرویس‌دهی REST
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   └── summarize.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── __init__.py
│   ├── dependencies.py
│   ├── exception_handlers.py
│   └── main.py
│
├── configs/                              # فایل‌های تنظیمات
│   ├── api_config.yaml
│   ├── cache_config.yaml
│   └── model_config.yaml
│
├── data/                                 # داده‌ها (gitignore برای حجیم‌ها)
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── eval/
│       └── .gitkeep
│
├── docs/                                 # مستندات فنی
│   ├── architecture.md
│   ├── evaluation_report.md
│   ├── installation_guide.md
│   ├── model_selection_report.md
│   └── future_extensions.md
│
├── models/                               # فایل‌های مدل (gitignore)
│   ├── base/
│   │   └── .gitkeep
│   ├── finetuned/
│   │   └── .gitkeep
│   └── quantized/                        # اضافه شد برای مدل‌های ONNX/GGUF
│       └── .gitkeep
│
├── scripts/                              # اسکریپت‌های اجرایی یک‌بار مصرف
│   ├── data/
│   │   ├── download_datasets.py
│   │   └── prepare_dataset.py
│   ├── training/
│   │   └── finetune.py
│   ├── quantization/
│   │   └── export_and_quantize.py        # هر دو ONNX و GGUF
│   └── evaluation/
│       ├── benchmark_speed.py
│       ├── compare_base_models.py
│       └── run_rouge_eval.py
│
├── src/                                  # هسته اصلی پروژه (کتابخانه)
│   └── arabic_summarizer/
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── cleaner.py
│       │   └── normalizer.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── base_engine.py
│       │   ├── onnx_engine.py
│       │   └── llamacpp_engine.py        # اضافه شد
│       ├── postprocessing/
│       │   ├── __init__.py
│       │   └── formatter.py
│       ├── evaluation/                   # اضافه شد
│       │   ├── __init__.py
│       │   ├── rouge_scorer.py
│       │   └── bert_scorer.py
│       ├── cache/
│       │   ├── __init__.py
│       │   └── cache_manager.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── config_loader.py
│       │   ├── hashing.py
│       │   └── logger.py
│       ├── __init__.py
│       ├── exceptions.py
│       ├── pipeline.py
│       └── summarizer.py
│
├── tests/
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_preprocessing.py
│   │   ├── test_postprocessing.py
│   │   ├── test_inference.py
│   │   ├── test_pipeline.py
│   │   ├── test_cache.py
│   │   └── test_hashing.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_api_endpoints.py
│   └── __init__.py
│
├── ui/
│   └── app_gradio.py
│
├── storage/                              # اضافه شد - داده‌های runtime
│   └── cache/
│       └── .gitkeep
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md