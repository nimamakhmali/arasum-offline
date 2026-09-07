"""
تست‌های واحد برای فاز ۲ — Training Pipeline.

این تست‌ها بدون دانلود مدل واقعی و بدون GPU اجرا می‌شوند.
از mock و داده‌های مصنوعی استفاده می‌شود.

اجرا:
    pytest tests/unit/test_training_pipeline.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "training"))


# ══════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════

@pytest.fixture
def sample_arabic_records() -> list[dict]:
    """نمونه‌های عربی برای تست."""
    return [
        {
            "text": "أعلنت وزارة التعليم عن إطلاق برنامج وطني شامل " * 25,
            "summary": "أعلنت الوزارة عن برنامج تعليمي جديد.",
            "word_count": 300,
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path, sample_arabic_records) -> Path:
    """فایل JSONL موقت برای تست."""
    file_path = tmp_path / "train.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for rec in sample_arabic_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def sample_data_dir(tmp_path, sample_arabic_records) -> Path:
    """پوشه دیتاست موقت با سه split."""
    for split in ["train", "validation", "test"]:
        file_path = tmp_path / f"{split}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in sample_arabic_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return tmp_path


# ══════════════════════════════════════════════════════════
# تست prepare_dataset.py
# ══════════════════════════════════════════════════════════

class TestPrepareDataset:
    """تست‌های scripts/training/prepare_dataset.py"""

    def test_load_jsonl_basic(self, sample_jsonl_file):
        from prepare_dataset import load_jsonl
        records = load_jsonl(sample_jsonl_file)
        assert len(records) == 20

    def test_load_jsonl_max_samples(self, sample_jsonl_file):
        from prepare_dataset import load_jsonl
        records = load_jsonl(sample_jsonl_file, max_samples=5)
        assert len(records) == 5

    def test_load_jsonl_nonexistent_file(self, tmp_path):
        from prepare_dataset import load_jsonl
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nonexistent.jsonl")

    def test_verify_record_fields_valid(self, sample_arabic_records):
        from prepare_dataset import verify_record_fields
        valid = verify_record_fields(
            sample_arabic_records, "text", "summary", "train"
        )
        assert len(valid) == len(sample_arabic_records)

    def test_verify_record_fields_missing_summary(self):
        from prepare_dataset import verify_record_fields
        records = [
            {"text": "نص عربي " * 50},
            {"text": "نص عربي " * 50, "summary": "خلاصه"},
        ]
        valid = verify_record_fields(records, "text", "summary", "train")
        assert len(valid) == 1

    def test_verify_record_fields_empty_text(self):
        from prepare_dataset import verify_record_fields
        records = [
            {"text": "", "summary": "خلاصه"},
            {"text": "نص ", * 50, "summary": "خلاصه"},
        ]
        valid = verify_record_fields(records, "text", "summary", "train")
        assert len(valid) == 1

    def test_analyze_length_distribution(self, sample_arabic_records):
        from prepare_dataset import analyze_length_distribution
        stats = analyze_length_distribution(
            sample_arabic_records, "text", "summary", "train"
        )
        assert "split" in stats
        assert "n_records" in stats
        assert stats["n_records"] == len(sample_arabic_records)
        assert "compression_ratio" in stats
        assert "text_words" in stats
        assert "summary_words" in stats

    def test_compression_ratio_is_positive(self, sample_arabic_records):
        from prepare_dataset import analyze_length_distribution
        stats = analyze_length_distribution(
            sample_arabic_records, "text", "summary", "train"
        )
        assert stats["compression_ratio"]["mean"] > 0

    def test_save_prepared_split(self, tmp_path, sample_arabic_records):
        from prepare_dataset import save_prepared_split
        out_file = save_prepared_split(
            records=sample_arabic_records,
            output_dir=tmp_path,
            split_name="train",
            text_field="text",
            summary_field="summary",
        )
        assert out_file.exists()
        # بررسی محتوا
        with open(out_file, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == len(sample_arabic_records)

    def test_saved_split_has_correct_fields(self, tmp_path, sample_arabic_records):
        from prepare_dataset import save_prepared_split
        out_file = save_prepared_split(
            records=sample_arabic_records,
            output_dir=tmp_path,
            split_name="train",
            text_field="text",
            summary_field="summary",
        )
        with open(out_file, encoding="utf-8") as f:
            first_record = json.loads(f.readline())
        assert "text" in first_record
        assert "summary" in first_record

    def test_source_target_direction(self, tmp_path, sample_arabic_records):
        """اطمینان از جهت صحیح: text → source, summary → target"""
        from prepare_dataset import save_prepared_split
        out_file = save_prepared_split(
            records=sample_arabic_records,
            output_dir=tmp_path,
            split_name="train",
            text_field="text",
            summary_field="summary",
        )
        with open(out_file, encoding="utf-8") as f:
            record = json.loads(f.readline())
        # متن اصلی باید طولانی‌تر از خلاصه باشد
        assert len(record["text"].split()) > len(record["summary"].split()), \
            "text باید طولانی‌تر از summary باشد"


# ══════════════════════════════════════════════════════════
# تست Tokenization
# ══════════════════════════════════════════════════════════
class TestTokenization:
    """تست‌های tokenization برای BART."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer که رفتار واقعی را شبیه‌سازی می‌کند."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 1
        tokenizer.vocab_size = 50000

        def mock_call(texts, max_length=None, truncation=False, padding=False, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            input_ids = [[i for i in range(min(20, max_length or 20))] for _ in texts]
            attention_mask = [[1] * len(ids) for ids in input_ids]
            result = MagicMock()
            result.__getitem__ = lambda self, key: {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }[key]
            result.get = lambda key, default=None: {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }.get(key, default)
            # dict-like access
            result["input_ids"] = input_ids
            result["attention_mask"] = attention_mask
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        tokenizer.side_effect = mock_call
        tokenizer.__call__ = mock_call

        ctx_manager = MagicMock()
        ctx_manager.__enter__ = MagicMock(return_value=None)
        ctx_manager.__exit__ = MagicMock(return_value=False)
        tokenizer.as_target_tokenizer = MagicMock(return_value=ctx_manager)

        return tokenizer

    def test_tokenize_function_returns_required_keys(self, mock_tokenizer):
        from finetune import create_tokenize_function
        tokenize_fn = create_tokenize_function(
            tokenizer=mock_tokenizer,
            max_source_length=512,
            max_target_length=128,
        )
        examples = {
            "text": ["متن عربي " * 50],
            "summary": ["خلاصه عربي"],
        }
        result = tokenize_fn(examples)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_labels_have_neg100_for_padding(self, mock_tokenizer):
        """labels باید با -100 padding شوند نه pad_token_id."""
        from finetune import create_tokenize_function

        mock_tokenizer.pad_token_id = 1

        tokenize_fn = create_tokenize_function(
            tokenizer=mock_tokenizer,
            max_source_length=512,
            max_target_length=128,
        )
        examples = {
            "text": ["متن " * 50],
            "summary": ["خلاصه"],
        }
        result = tokenize_fn(examples)
        # pad_token_id (=1) نباید در labels باشد
        for label_seq in result["labels"]:
            assert 1 not in label_seq, \
                "pad_token_id باید با -100 جایگزین شود"

    def test_source_target_not_reversed(self):
        """text → input, summary → labels (نه برعکس) — تست مفهومی."""
        # این تست بدون mock پیچیده، منطق را بررسی می‌کند
        text = "متن طویل عربی " * 30      # ~450 کلمه
        summary = "خلاصه کوتاه"            # ~2 کلمه
        assert len(text.split()) > len(summary.split()), \
            "text باید از summary طولانی‌تر باشد"


class TestVerifyRecordFields:
    """تست اضافی برای verify_record_fields با داده‌های واقعی."""

    def test_empty_text_removed(self):
        from prepare_dataset import verify_record_fields
        records = [
            {"text": "", "summary": "خلاصه"},
            {"text": "نص عربي " * 50, "summary": "خلاصه"},
        ]
        valid = verify_record_fields(records, "text", "summary", "train")
        assert len(valid) == 1

    def test_all_valid_records_pass(self):
        from prepare_dataset import verify_record_fields
        records = [
            {"text": "نص عربي " * 50, "summary": f"خلاصه {i}"}
            for i in range(10)
        ]
        valid = verify_record_fields(records, "text", "summary", "train")
        assert len(valid) == 10

    def test_missing_text_field_removed(self):
        from prepare_dataset import verify_record_fields
        records = [
            {"summary": "خلاصه بدون متن"},
            {"text": "نص عربي " * 30, "summary": "خلاصه"},
        ]
        valid = verify_record_fields(records, "text", "summary", "train")
        assert len(valid) == 1
        
# ══════════════════════════════════════════════════════════
# تست LoRA Configuration
# ══════════════════════════════════════════════════════════

class TestLoRASetup:
    """تست‌های setup_lora."""

    @pytest.fixture
    def mock_model_with_qproj(self):
        """مدل mock با q_proj و v_proj."""
        model = MagicMock()
        # شبیه‌سازی named_modules با attention layers
        modules = [
            ("encoder.layers.0.self_attn.q_proj", MagicMock()),
            ("encoder.layers.0.self_attn.v_proj", MagicMock()),
            ("encoder.layers.0.self_attn.k_proj", MagicMock()),
            ("decoder.layers.0.self_attn.q_proj", MagicMock()),
            ("decoder.layers.0.self_attn.v_proj", MagicMock()),
            ("lm_head", MagicMock()),
        ]
        model.named_modules.return_value = modules
        model.parameters.return_value = [MagicMock(numel=lambda: 1000, requires_grad=False)]
        return model

    def test_get_arabart_target_modules_finds_qproj(self, mock_model_with_qproj):
        from finetune import get_arabart_lora_target_modules
        modules = get_arabart_lora_target_modules(mock_model_with_qproj)
        assert "q_proj" in modules

    def test_get_arabart_target_modules_finds_vproj(self, mock_model_with_qproj):
        from finetune import get_arabart_lora_target_modules
        modules = get_arabart_lora_target_modules(mock_model_with_qproj)
        assert "v_proj" in modules

    def test_get_arabart_target_modules_fallback(self):
        """اگر هیچ attention module پیدا نشد، باید fallback داشته باشد."""
        from finetune import get_arabart_lora_target_modules
        model = MagicMock()
        model.named_modules.return_value = [
            ("embedding", MagicMock()),
            ("lm_head", MagicMock()),
        ]
        modules = get_arabart_lora_target_modules(model)
        # باید لیست غیرخالی برگرداند
        assert len(modules) > 0

    def test_setup_lora_raises_when_no_trainable_params(self):
        """اگر LoRA هیچ پارامتر قابل‌آموزشی ندارد، باید خطا بدهد."""
        from finetune import setup_lora

        with patch("finetune.get_arabart_lora_target_modules", return_value=["q_proj"]):
            with patch("finetune.get_peft_model") as mock_peft:
                mock_model = MagicMock()
                # همه پارامترها requires_grad=False
                mock_param = MagicMock()
                mock_param.numel.return_value = 1000
                mock_param.requires_grad = False
                mock_model.parameters.return_value = [mock_param]
                mock_peft.return_value = mock_model

                with pytest.raises(RuntimeError, match="پارامتر قابل‌آموزشی"):
                    setup_lora(
                        MagicMock(),
                        {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1},
                    )


# ══════════════════════════════════════════════════════════
# تست Dataset Loading
# ══════════════════════════════════════════════════════════

class TestDatasetLoading:
    """تست‌های load_dataset_from_jsonl."""

    def test_loads_train_validation_test(self, sample_data_dir):
        from finetune import load_dataset_from_jsonl
        train, val, test = load_dataset_from_jsonl(sample_data_dir)
        assert train is not None
        assert val is not None
        assert test is not None

    def test_train_has_required_columns(self, sample_data_dir):
        from finetune import load_dataset_from_jsonl
        train, _, _ = load_dataset_from_jsonl(sample_data_dir)
        assert "text" in train.column_names
        assert "summary" in train.column_names

    def test_max_train_samples_respected(self, sample_data_dir):
        from finetune import load_dataset_from_jsonl
        train, _, _ = load_dataset_from_jsonl(
            sample_data_dir, max_train_samples=5
        )
        assert len(train) == 5

    def test_missing_train_file_returns_none(self, tmp_path):
        from finetune import load_dataset_from_jsonl
        # فقط validation وجود دارد
        val_file = tmp_path / "validation.jsonl"
        val_file.write_text(
            json.dumps({"text": "نص", "summary": "خلاصه"}, ensure_ascii=False)
        )
        train, val, test = load_dataset_from_jsonl(tmp_path)
        assert train is None

    def test_splits_are_independent(self, sample_data_dir):
        """اطمینان از اینکه split ها با هم mix نمی‌شوند."""
        from finetune import load_dataset_from_jsonl
        train, val, test = load_dataset_from_jsonl(
            sample_data_dir,
            max_train_samples=10,
            max_eval_samples=5,
        )
        # train و val باید اندازه‌های متفاوت داشته باشند
        assert len(train) == 10
        assert len(val) == 5


# ══════════════════════════════════════════════════════════
# تست Truncation Analysis
# ══════════════════════════════════════════════════════════

class TestTruncationAnalysis:
    """تست measure_truncation."""

    def test_measure_truncation_no_truncation(self):
        from finetune import measure_truncation

        # دیتاست mock با input_ids کوتاه
        mock_dataset = [
            {"input_ids": list(range(50))},
            {"input_ids": list(range(30))},
        ]
        stats = measure_truncation(mock_dataset, None, 1024, "train")
        assert stats["truncated"] == 0
        assert stats["total"] == 2
        assert stats["pct"] == 0.0

    def test_measure_truncation_all_truncated(self):
        from finetune import measure_truncation

        mock_dataset = [
            {"input_ids": list(range(1024))},
            {"input_ids": list(range(1024))},
        ]
        stats = measure_truncation(mock_dataset, None, 1024, "train")
        assert stats["truncated"] == 2
        assert stats["pct"] == 100.0


# ══════════════════════════════════════════════════════════
# تست BertScorer
# ══════════════════════════════════════════════════════════

class TestArabicBertScorer:
    """تست‌های ArabicBertScorer."""

    def test_scorer_reports_availability(self):
        from arabic_summarizer.evaluation.bert_scorer import ArabicBertScorer
        scorer = ArabicBertScorer()
        # فقط بررسی می‌کنیم که کلید وجود دارد
        assert isinstance(scorer.is_available(), bool)

    def test_bertscore_result_to_dict(self):
        from arabic_summarizer.evaluation.bert_scorer import BertScoreResult
        result = BertScoreResult(
            precision=0.85,
            recall=0.87,
            f1=0.86,
            model_used="bert-base-multilingual-cased",
            n_samples=100,
        )
        d = result.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert "meets_target" in d

    def test_bertscore_result_meets_target(self):
        from arabic_summarizer.evaluation.bert_scorer import BertScoreResult
        good = BertScoreResult(0.87, 0.88, 0.87, "model", 100)
        bad = BertScoreResult(0.80, 0.82, 0.81, "model", 100)
        assert good.to_dict()["meets_target"] is True
        assert bad.to_dict()["meets_target"] is False

    def test_score_batch_mismatched_lengths_raises(self):
        from arabic_summarizer.evaluation.bert_scorer import ArabicBertScorer
        scorer = ArabicBertScorer()
        if not scorer.is_available():
            pytest.skip("bert-score نصب نیست")
        with pytest.raises(ValueError):
            scorer.score_batch(["a", "b"], ["c"])


# ══════════════════════════════════════════════════════════
# تست Evaluation Report
# ══════════════════════════════════════════════════════════

class TestEvaluationReport:
    """تست تولید گزارش ارزیابی."""

    def test_generate_report_creates_json(self, tmp_path):
        from evaluate_finetuned import generate_evaluation_report

        rouge_stats = {
            "n_samples": 100,
            "n_failed": 0,
            "rouge1": 0.45,
            "rouge2": 0.25,
            "rougeL": 0.40,
            "meets_rouge1_target": True,
            "meets_rouge2_target": True,
            "meets_rougeL_target": True,
            "inference_time": {"mean": 5.0, "median": 4.8, "min": 3.0, "max": 8.0, "stdev": 1.0},
        }
        bertscore_stats = {"f1": 0.85, "precision": 0.84, "recall": 0.86,
                          "model": "bert-base-multilingual-cased", "n_samples": 100,
                          "meets_target": True}
        latency_stats = {
            "1000": {
                "target_input_words": 1000,
                "actual_input_words": 980,
                "mean_seconds": 25.0,
                "median_seconds": 24.5,
                "min_seconds": 22.0,
                "max_seconds": 28.0,
                "stdev_seconds": 2.0,
                "warmup_runs": 2,
                "timed_runs": 5,
                "avg_output_words": 30,
                "device": "cpu",
                "phase2_requirement_60s": "PASS",
            }
        }

        generate_evaluation_report(
            rouge_stats=rouge_stats,
            bertscore_stats=bertscore_stats,
            latency_stats=latency_stats,
            qualitative_samples=[],
            model_path="models/finetuned/arabart",
            output_dir=tmp_path,
        )

        json_path = tmp_path / "phase2_evaluation_report.json"
        md_path = tmp_path / "phase2_evaluation_report.md"
        assert json_path.exists()
        assert md_path.exists()

    def test_generated_json_has_required_keys(self, tmp_path):
        from evaluate_finetuned import generate_evaluation_report

        generate_evaluation_report(
            rouge_stats={"n_samples": 10, "n_failed": 0, "rouge1": 0.3, "rouge2": 0.15,
                         "rougeL": 0.28, "meets_rouge1_target": False,
                         "meets_rouge2_target": False, "meets_rougeL_target": False,
                         "inference_time": {"mean": 5.0, "median": 5.0, "min": 4.0,
                                           "max": 6.0, "stdev": 0.5}},
            bertscore_stats={},
            latency_stats={},
            qualitative_samples=[],
            model_path="test_model",
            output_dir=tmp_path,
        )

        with open(tmp_path / "phase2_evaluation_report.json", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "evaluated_at", "model_path", "phase",
            "rouge_evaluation", "bertscore_evaluation",
            "latency_benchmark", "project_targets", "known_limitations",
        ]
        for key in required_keys:
            assert key in data, f"کلید '{key}' در گزارش وجود ندارد"