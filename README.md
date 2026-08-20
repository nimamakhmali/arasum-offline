# arabic-offline-summarizer
An offline, high-accuracy Arabic text summarization engine optimized for real-world applications, fast local execution, and complete internet independence.


#  Arabic Offline Summarizer (AOS) v2.0

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-Apache%202.0-orange)
![Offline](https://img.shields.io/badge/mode-100%25%20Offline-success)
![Arabic](https://img.shields.io/badge/language-Arabic%20MSA-red)

**Smart Arabic Text Summarizer – Fully Offline, Fast, Accurate**

*Intelligent Arabic Text Summarization - Fully Offline, Fast, Accurate*

</div>

---

## Key Features

| Feature | Details |
|-------|---------|
| 🔌 **Fully Offline** | No internet required after installation |
| ⚡ **Fast** | 5–15 seconds for standard texts |
| 🎯 **Accurate** | 10–30% summary while retaining core concepts |
| 🔄 **Hybrid** | Extractive + Abstractive combination |
| 📦 **ONNX** | Optimized and quantized model |
| 🌐 **Versatile** | Quran, news, documents, research |
---

## 🚀 Quick Setup

```bash
# 1. Clone the project
git clone https://github.com/your-org/arabic-offline-summarizer.git
cd arabic-offline-summarizer

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model (one-time - requires internet)
python scripts/download_model.py

# 5. Quick test
python run_demo.py --quick
```


---

## Design

```
      input
        │
        ▼
┌───────────────────┐
│   Preprocessor    │ ← Normalization, cleaning, formation
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Extractive Layer │  ← TF-IDF + Position (for long texts)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  AraT5v2 (ONNX)   │ ← Abstractive summarization
│  INT8 Quantized   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Postprocessor    │  ← Removing repetition, correcting the writing
└───────────────────┘
        │
        ▼
     summarize
```

---

## structure

```
arabic_offline_summarizer/
├── src/arabic_summarizer/
│   ├── __init__.py        
│   ├── core.py            
│   ├── preprocessor.py     
│   ├── postprocessor.py     
│   ├── extractive.py       
│   ├── chunker.py          
│   ├── evaluate.py         
│   └── config.py           
├── demo/
│   └── app.py               
├── scripts/
│   ├── download_model.py  
│   └── export_to_onnx.py   
├── tests/
│   └── test_summarizer.py  
├── models/                 
├── run_demo.py             
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## (ONNX)

```bash

python scripts/download_model.py

python scripts/export_to_onnx.py

python scripts/export_to_onnx.py --benchmark

python run_demo.py --mode onnx
```

---

## CLI

```bash
python demo/app.py
# http://localhost:7860
```

---

##  Evaluation

```python
from arabic_summarizer import ArabicSummarizationEvaluator

evaluator = ArabicSummarizationEvaluator()

result = evaluator.evaluate_single(
    original=original_text,
    generated=summary,
    reference=human_summary  
)
print(f"Quality: {result['quality_score']}/100")
print(f"ROUGE-1: {result.get('rouge1', 'N/A')}")

report = evaluator.benchmark_summarizer(summarizer, test_cases)
```

---

##  tests

```bash
pytest tests/ -v

pytest tests/ -v -m offline

pytest tests/ --cov=arabic_summarizer --cov-report=html
```

---

##  کاربردها

- **🕌 Quran & Exegesis**: Memorization support + precise summaries
- **📰 Media & News**: Quick summaries of Arabic news
- **📚 Research**: Summaries of scholarly articles and research
- **📋 Official Documents**: Summaries of contracts and directives
- **🎓 Education**: Summaries of course materials
---

##  License

Apache License 2.0 — Free for commercial and non-commercial use

---

##  Contact
nimamakhmali2004@gmail.com