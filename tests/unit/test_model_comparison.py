"""
Unit Tests برای TASK-102: مقایسه مدل‌های پایه.

تمام تست‌ها بدون اینترنت و بدون دانلود مدل واقعی اجرا می‌شوند.
از mock و fake objects استفاده می‌شود.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# import مستقیم ماژول‌های تست‌شونده
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arabic_summarizer.evaluation.rouge_scorer import (
    ArabicRougeScorer,
    RougeScores,
    _normalize_arabic_for_rouge,
)

# import اسکریپت مقایسه
_SCRIPTS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "model_selection"
sys.path.insert(0, str(_SCRIPTS_PATH))

from compare_base_models import (
    BenchmarkRunner,
    ModelBenchmarkResult,
    ModelConfig,
    ModelRunner,
    SampleResult,
    compute_ranking_score,
    load_benchmark_samples,
    save_csv_summary,
    save_json_report,
    select_recommended_model,
)


# Fixtures


@pytest.fixture
def sample_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="moussaKam/AraBART",
        model_name="AraBART",
        architecture="BART",
        model_type="seq2seq",
        tokenizer_id="moussaKam/AraBART",
        max_input_length=1024,
        enabled=True,
        priority=1,
        arabic_optimized=True,
        task_prefix=None,
        notes="تست",
        generation_params_override={},
    )


@pytest.fixture
def sample_model_config_t5() -> ModelConfig:
    return ModelConfig(
        model_id="UBC-NLP/AraT5",
        model_name="AraT5",
        architecture="T5",
        model_type="seq2seq",
        tokenizer_id="UBC-NLP/AraT5",
        max_input_length=512,
        enabled=True,
        priority=2,
        arabic_optimized=True,
        task_prefix="summarize: ",
        notes="تست",
        generation_params_override={"max_new_tokens": 150},
    )


@pytest.fixture
def arabic_samples() -> list[dict]:
    """نمونه‌های عربی برای تست."""
    return [
        {
            "id": f"sample_{i}",
            "text": "أعلنت وزارة التعليم عن إطلاق برنامج وطني شامل " * 20,
            "summary": "أعلنت الوزارة عن برنامج تعليمي جديد.",
        }
        for i in range(5)
    ]


@pytest.fixture
def successful_result(sample_model_config) -> ModelBenchmarkResult:
    """یک نتیجه موفق برای تست."""
    samples = [
        SampleResult(
            sample_id=f"s{i}",
            input_text="نص عربي " * 50,
            reference_summary="خلاصه مرجع",
            generated_summary="خلاصه تولیدشده",
            inference_time_seconds=2.0 + i * 0.5,
            preprocessing_time_seconds=0.1,
            rouge_scores=RougeScores(rouge1=0.45, rouge2=0.25, rougeL=0.40),
            status="success",
        )
        for i in range(5)
    ]
    return ModelBenchmarkResult(
        model_id=sample_model_config.model_id,
        model_name=sample_model_config.model_name,
        architecture=sample_model_config.architecture,
        status="success",
        error_message=None,
        load_time_seconds=5.0,
        sample_results=samples,
        parameter_count=139_000_000,
    )


@pytest.fixture
def failed_result(sample_model_config_t5) -> ModelBenchmarkResult:
    """یک نتیجه ناموفق برای تست."""
    return ModelBenchmarkResult(
        model_id=sample_model_config_t5.model_id,
        model_name=sample_model_config_t5.model_name,
        architecture=sample_model_config_t5.architecture,
        status="failed",
        error_message="خطای تست",
        load_time_seconds=0.0,
    )


# تست RougeScores


class TestRougeScores:

    def test_to_dict_keys(self):
        scores = RougeScores(rouge1=0.45, rouge2=0.25, rougeL=0.40)
        d = scores.to_dict()
        assert set(d.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_to_dict_values_rounded(self):
        scores = RougeScores(rouge1=0.123456, rouge2=0.234567, rougeL=0.345678)
        d = scores.to_dict()
        assert d["rouge1"] == 0.1235
        assert d["rouge2"] == 0.2346
        assert d["rougeL"] == 0.3457

    def test_average_empty_returns_zeros(self):
        avg = RougeScores.average([])
        assert avg.rouge1 == 0.0
        assert avg.rouge2 == 0.0
        assert avg.rougeL == 0.0

    def test_average_single(self):
        scores = [RougeScores(rouge1=0.5, rouge2=0.3, rougeL=0.4)]
        avg = RougeScores.average(scores)
        assert avg.rouge1 == pytest.approx(0.5)
        assert avg.rouge2 == pytest.approx(0.3)
        assert avg.rougeL == pytest.approx(0.4)

    def test_average_multiple(self):
        scores = [
            RougeScores(rouge1=0.4, rouge2=0.2, rougeL=0.3),
            RougeScores(rouge1=0.6, rouge2=0.4, rougeL=0.5),
        ]
        avg = RougeScores.average(scores)
        assert avg.rouge1 == pytest.approx(0.5)
        assert avg.rouge2 == pytest.approx(0.3)
        assert avg.rougeL == pytest.approx(0.4)


# تست normalize


class TestNormalizeArabicForRouge:

    def test_removes_diacritics(self):
        result = _normalize_arabic_for_rouge("كَتَبَ")
        assert "َ" not in result
        assert "كتب" in result

    def test_removes_tatweel(self):
        text = "ك\u0640ت\u0640اب"
        result = _normalize_arabic_for_rouge(text)
        assert "\u0640" not in result

    def test_normalizes_whitespace(self):
        result = _normalize_arabic_for_rouge("كلمة   أخرى")
        assert "  " not in result

    def test_empty_string(self):
        assert _normalize_arabic_for_rouge("") == ""

    def test_deterministic(self):
        text = "الحَمْدُ لِلَّهِ"
        assert _normalize_arabic_for_rouge(text) == _normalize_arabic_for_rouge(text)


# تست ArabicRougeScorer


class TestArabicRougeScorer:

    @pytest.fixture
    def scorer(self):
        return ArabicRougeScorer()

    def test_perfect_match_returns_high_scores(self, scorer):
        text = "أعلنت وزارة التعليم عن إطلاق برنامج وطني"
        scores = scorer.score(text, text)
        assert scores.rouge1 > 0.9
        assert scores.rougeL > 0.9

    def test_empty_hypothesis_returns_zeros(self, scorer):
        scores = scorer.score("", "مرجع")
        assert scores.rouge1 == 0.0

    def test_empty_reference_returns_zeros(self, scorer):
        scores = scorer.score("خروجی", "")
        assert scores.rouge1 == 0.0

    def test_score_batch_length_mismatch_raises(self, scorer):
        with pytest.raises(ValueError):
            scorer.score_batch(["a"], ["b", "c"])

    def test_score_batch_returns_correct_length(self, scorer):
        hyps = ["خلاصه اول", "خلاصه دوم"]
        refs = ["مرجع اول", "مرجع دوم"]
        individual, aggregate = scorer.score_batch(hyps, refs)
        assert len(individual) == 2
        assert isinstance(aggregate, RougeScores)

    def test_scores_between_zero_and_one(self, scorer):
        scores = scorer.score("أعلنت وزارة", "أعلنت الوزارة عن برنامج")
        assert 0.0 <= scores.rouge1 <= 1.0
        assert 0.0 <= scores.rouge2 <= 1.0
        assert 0.0 <= scores.rougeL <= 1.0


# تست ModelBenchmarkResult


class TestModelBenchmarkResult:

    def test_successful_samples_filter(self, successful_result):
        assert len(successful_result.successful_samples) == 5
        assert len(successful_result.failed_samples) == 0

    def test_aggregate_rouge_average(self, successful_result):
        rouge = successful_result.aggregate_rouge
        assert rouge is not None
        assert rouge.rouge1 == pytest.approx(0.45)

    def test_inference_times(self, successful_result):
        times = successful_result.inference_times
        assert len(times) == 5

    def test_avg_inference_time(self, successful_result):
        avg = successful_result.avg_inference_time
        assert avg is not None
        assert avg > 0

    def test_median_inference_time(self, successful_result):
        median = successful_result.median_inference_time
        assert median is not None
        assert median > 0

    def test_failed_result_no_samples(self, failed_result):
        assert len(failed_result.successful_samples) == 0
        assert failed_result.aggregate_rouge is None
        assert failed_result.avg_inference_time is None

    def test_to_summary_dict_structure(self, successful_result):
        d = successful_result.to_summary_dict()
        assert "model_id" in d
        assert "model_name" in d
        assert "status" in d
        assert "metrics" in d
        assert "performance" in d
        assert "samples" in d

    def test_partial_status_when_some_fail(self, sample_model_config):
        samples = [
            SampleResult(
                sample_id="s1",
                input_text="نص",
                reference_summary="مرجع",
                generated_summary="خلاصه",
                inference_time_seconds=1.0,
                preprocessing_time_seconds=0.1,
                rouge_scores=RougeScores(0.4, 0.2, 0.3),
                status="success",
            ),
            SampleResult(
                sample_id="s2",
                input_text="نص",
                reference_summary=None,
                generated_summary="",
                inference_time_seconds=0.0,
                preprocessing_time_seconds=0.0,
                rouge_scores=None,
                status="failed",
                error_message="خطا",
            ),
        ]
        result = ModelBenchmarkResult(
            model_id="test",
            model_name="Test",
            architecture="BART",
            status="partial",
            error_message=None,
            load_time_seconds=1.0,
            sample_results=samples,
        )
        assert len(result.successful_samples) == 1
        assert len(result.failed_samples) == 1


# تست compute_ranking_score


class TestComputeRankingScore:

    @pytest.fixture
    def weights(self):
        return {"rouge1": 0.25, "rouge2": 0.35, "rougeL": 0.30, "speed": 0.10}

    def _make_result(self, r1, r2, rL, avg_time, name="Model") -> ModelBenchmarkResult:
        samples = [
            SampleResult(
                sample_id="s1",
                input_text="نص",
                reference_summary="مرجع",
                generated_summary="خلاصه",
                inference_time_seconds=avg_time,
                preprocessing_time_seconds=0.1,
                rouge_scores=RougeScores(r1, r2, rL),
                status="success",
            )
        ]
        return ModelBenchmarkResult(
            model_id=f"test/{name}",
            model_name=name,
            architecture="BART",
            status="success",
            error_message=None,
            load_time_seconds=1.0,
            sample_results=samples,
        )

    def test_failed_model_scores_zero(self, weights):
        failed = ModelBenchmarkResult(
            model_id="x",
            model_name="X",
            architecture="T5",
            status="failed",
            error_message="err",
            load_time_seconds=0.0,
        )
        score = compute_ranking_score(failed, weights, [failed])
        assert score == 0.0

    def test_single_model_scores_one(self, weights):
        result = self._make_result(0.5, 0.3, 0.4, 2.0)
        score = compute_ranking_score(result, weights, [result])
        assert score == pytest.approx(1.0)

    def test_better_rouge_higher_score(self, weights):
        good = self._make_result(0.6, 0.4, 0.5, 2.0, "Good")
        bad = self._make_result(0.3, 0.15, 0.25, 2.0, "Bad")
        all_results = [good, bad]
        score_good = compute_ranking_score(good, weights, all_results)
        score_bad = compute_ranking_score(bad, weights, all_results)
        assert score_good > score_bad

    def test_faster_model_higher_score_when_rouge_equal(self, weights):
        fast = self._make_result(0.5, 0.3, 0.4, 1.0, "Fast")
        slow = self._make_result(0.5, 0.3, 0.4, 10.0, "Slow")
        all_results = [fast, slow]
        score_fast = compute_ranking_score(fast, weights, all_results)
        score_slow = compute_ranking_score(slow, weights, all_results)
        assert score_fast > score_slow

    def test_deterministic(self, weights):
        result = self._make_result(0.5, 0.3, 0.4, 2.0)
        score1 = compute_ranking_score(result, weights, [result])
        score2 = compute_ranking_score(result, weights, [result])
        assert score1 == score2


# تست select_recommended_model


class TestSelectRecommendedModel:

    @pytest.fixture
    def weights(self):
        return {"rouge1": 0.25, "rouge2": 0.35, "rougeL": 0.30, "speed": 0.10}

    def test_returns_none_when_all_failed(self, weights):
        failed = ModelBenchmarkResult(
            model_id="x",
            model_name="X",
            architecture="BART",
            status="failed",
            error_message="err",
            load_time_seconds=0.0,
        )
        result = select_recommended_model([failed], weights)
        assert result is None

    def test_returns_best_model(self, weights):
        def _make(name, r1, r2, rL, time):
            samples = [
                SampleResult(
                    sample_id="s1",
                    input_text="نص",
                    reference_summary="مرجع",
                    generated_summary="خلاصه",
                    inference_time_seconds=time,
                    preprocessing_time_seconds=0.1,
                    rouge_scores=RougeScores(r1, r2, rL),
                    status="success",
                )
            ]
            return ModelBenchmarkResult(
                model_id=f"test/{name}",
                model_name=name,
                architecture="BART",
                status="success",
                error_message=None,
                load_time_seconds=1.0,
                sample_results=samples,
            )

        good = _make("AraBART", 0.6, 0.4, 0.5, 2.0)
        bad = _make("mT5", 0.3, 0.15, 0.25, 5.0)

        result = select_recommended_model([good, bad], weights)
        assert result is not None
        assert result["model_name"] == "AraBART"

    def test_result_has_required_keys(self, weights, successful_result):
        result = select_recommended_model([successful_result], weights)
        assert result is not None
        assert "model_id" in result
        assert "model_name" in result
        assert "ranking_score" in result
        assert "reason" in result
        assert "all_rankings" in result

    def test_benchmark_continues_after_one_failure(self, weights, successful_result, failed_result):
        """benchmark باید بعد از failure یک مدل ادامه دهد."""
        result = select_recommended_model([failed_result, successful_result], weights)
        assert result is not None
        assert result["model_name"] == successful_result.model_name


# تست load_benchmark_samples


class TestLoadBenchmarkSamples:

    def test_loads_from_jsonl_file(self, tmp_path):
        data_file = tmp_path / "test.jsonl"
        records = [
            {"id": f"s{i}", "text": "نص عربي " * 50, "summary": "خلاصه"}
            for i in range(30)
        ]
        with open(data_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        samples = load_benchmark_samples(
            dataset_path=data_file, num_samples=20, seed=42
        )
        assert len(samples) == 20

    def test_deterministic_with_same_seed(self, tmp_path):
        data_file = tmp_path / "test.jsonl"
        records = [{"id": f"s{i}", "text": f"نص {i} " * 50} for i in range(50)]
        with open(data_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        samples1 = load_benchmark_samples(data_file, 20, seed=42)
        samples2 = load_benchmark_samples(data_file, 20, seed=42)
        assert [s["id"] for s in samples1] == [s["id"] for s in samples2]

    def test_different_seed_different_order(self, tmp_path):
        data_file = tmp_path / "test.jsonl"
        records = [{"id": f"s{i}", "text": f"نص {i} " * 50} for i in range(50)]
        with open(data_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        samples1 = load_benchmark_samples(data_file, 20, seed=42)
        samples2 = load_benchmark_samples(data_file, 20, seed=99)
        assert [s["id"] for s in samples1] != [s["id"] for s in samples2]

    def test_returns_empty_when_file_not_found(self):
        samples = load_benchmark_samples(
            dataset_path=Path("/nonexistent/path.jsonl"),
            num_samples=10,
            seed=42,
        )
        assert samples == []

    def test_sample_ids_in_output(self, tmp_path):
        data_file = tmp_path / "test.jsonl"
        records = [{"id": f"id_{i}", "text": "نص " * 50} for i in range(10)]
        with open(data_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        samples = load_benchmark_samples(data_file, 5, seed=42)
        for s in samples:
            assert "id" in s
            assert "text" in s


# تست save_json_report و save_csv_summary

class TestOutputWriters:

    @pytest.fixture
    def metadata(self):
        return {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "device": "cpu",
            "num_samples_requested": 20,
            "num_samples_actual": 20,
            "seed": 42,
            "dataset_source": "test",
            "generation_config": {},
            "ranking_weights": {},
        }

    def test_save_json_report_creates_file(
        self, tmp_path, successful_result, failed_result, metadata
    ):
        recommended = {"model_id": "x", "model_name": "X", "ranking_score": 0.8, "reason": "test", "all_rankings": []}
        path = save_json_report(
            [successful_result, failed_result], recommended, metadata, tmp_path
        )
        assert path.exists()

    def test_save_json_report_valid_json(
        self, tmp_path, successful_result, metadata
    ):
        path = save_json_report([successful_result], None, metadata, tmp_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "benchmark_metadata" in data
        assert "models" in data
        assert "recommended_model" in data

    def test_save_json_report_contains_all_models(
        self, tmp_path, successful_result, failed_result, metadata
    ):
        path = save_json_report(
            [successful_result, failed_result], None, metadata, tmp_path
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model_names = [m["model_name"] for m in data["models"]]
        assert successful_result.model_name in model_names
        assert failed_result.model_name in model_names

    def test_save_csv_summary_creates_file(
        self, tmp_path, successful_result, failed_result
    ):
        path = save_csv_summary([successful_result, failed_result], tmp_path)
        assert path.exists()

    def test_save_csv_summary_has_correct_columns(
        self, tmp_path, successful_result
    ):
        import csv as csv_module

        path = save_csv_summary([successful_result], tmp_path)
        with open(path, encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            headers = reader.fieldnames
        assert "model_id" in headers
        assert "rouge1" in headers
        assert "avg_inference_time" in headers
        assert "status" in headers

    def test_save_csv_has_correct_row_count(
        self, tmp_path, successful_result, failed_result
    ):
        import csv as csv_module

        path = save_csv_summary([successful_result, failed_result], tmp_path)
        with open(path, encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2