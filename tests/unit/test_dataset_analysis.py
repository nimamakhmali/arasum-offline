"""
Unit Tests برای تحلیل دیتاست.
بدون اینترنت و بدون فایل واقعی اجرا می‌شوند.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "data"))

from analyze_dataset import (
    _arabic_ratio,
    _bin_label,
    _compression_ratio,
    _count_words,
    _percentile,
    analyze_split,
    generate_markdown_report,
    load_jsonl,
)


# ══════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════


def _make_record(text_words: int, summary_words: int) -> dict:
    """رکورد ساختگی با تعداد کلمه مشخص."""
    text = " ".join(["كلمة"] * text_words)
    summary = " ".join(["ملخص"] * summary_words) if summary_words > 0 else ""
    return {"id": f"r_{text_words}", "text": text, "summary": summary}


@pytest.fixture
def sample_records():
    return [
        _make_record(500, 75),
        _make_record(800, 120),
        _make_record(1200, 200),
        _make_record(200, 30),   # خارج از محدوده - کوتاه
        _make_record(3500, 400), # خارج از محدوده - بلند
    ]


# ══════════════════════════════════════════════════════════
# تست توابع کمکی
# ══════════════════════════════════════════════════════════


class TestHelperFunctions:

    def test_count_words_basic(self):
        assert _count_words("كلمة أخرى ثالثة") == 3

    def test_count_words_empty(self):
        assert _count_words("") == 0

    def test_count_words_single(self):
        assert _count_words("كلمة") == 1

    def test_arabic_ratio_pure_arabic(self):
        ratio = _arabic_ratio("كلمة عربية")
        assert ratio > 0.9

    def test_arabic_ratio_empty(self):
        assert _arabic_ratio("") == 0.0

    def test_compression_ratio_basic(self):
        assert _compression_ratio(100, 20) == pytest.approx(0.2)

    def test_compression_ratio_zero_text(self):
        assert _compression_ratio(0, 10) == 0.0

    def test_percentile_median(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == pytest.approx(3.0)

    def test_percentile_min(self):
        values = [1.0, 2.0, 3.0]
        assert _percentile(values, 0) == pytest.approx(1.0)

    def test_percentile_max(self):
        values = [1.0, 2.0, 3.0]
        assert _percentile(values, 100) == pytest.approx(3.0)

    def test_bin_label_in_range(self):
        label = _bin_label(500)
        assert "500" in label or "متوسط" in label

    def test_bin_label_out_of_range(self):
        label = _bin_label(4000)
        assert "3000" in label or "خارج" in label


# ══════════════════════════════════════════════════════════
# تست analyze_split
# ══════════════════════════════════════════════════════════


class TestAnalyzeSplit:

    def test_empty_records(self):
        result = analyze_split([], "train")
        assert result["count"] == 0

    def test_total_records_correct(self, sample_records):
        result = analyze_split(sample_records, "train")
        assert result["total_records"] == 5

    def test_in_range_count(self, sample_records):
        result = analyze_split(sample_records, "train")
        # 500، 800، 1200 در محدوده هستند
        assert result["in_range_300_3000"] == 3

    def test_in_range_percent(self, sample_records):
        result = analyze_split(sample_records, "train")
        assert result["in_range_percent"] == pytest.approx(60.0)

    def test_has_summary_count(self, sample_records):
        result = analyze_split(sample_records, "train")
        assert result["has_summary"] == 5

    def test_text_length_mean(self, sample_records):
        result = analyze_split(sample_records, "train")
        tl = result["text_length"]
        expected_mean = (500 + 800 + 1200 + 200 + 3500) / 5
        assert tl["mean"] == pytest.approx(expected_mean, rel=0.01)

    def test_text_length_min_max(self, sample_records):
        result = analyze_split(sample_records, "train")
        tl = result["text_length"]
        assert tl["min"] == 200
        assert tl["max"] == 3500

    def test_summary_length_present(self, sample_records):
        result = analyze_split(sample_records, "train")
        assert "summary_length" in result

    def test_compression_ratio_present(self, sample_records):
        result = analyze_split(sample_records, "train")
        assert "compression_ratio" in result

    def test_compression_ratio_values(self, sample_records):
        result = analyze_split(sample_records, "train")
        cr = result["compression_ratio"]
        assert 0.0 < cr["mean"] < 1.0

    def test_length_distribution_keys(self, sample_records):
        result = analyze_split(sample_records, "train")
        dist = result["length_distribution"]
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_no_summary_records(self):
        records = [
            {"id": "1", "text": " ".join(["كلمة"] * 500), "summary": ""},
            {"id": "2", "text": " ".join(["كلمة"] * 600)},
        ]
        result = analyze_split(records, "train")
        assert result["has_summary"] == 0
        assert "summary_length" not in result

    def test_arabic_ratio_computed(self, sample_records):
        result = analyze_split(sample_records, "train")
        ar = result["arabic_ratio"]
        assert "mean" in ar
        assert 0.0 <= ar["mean"] <= 1.0

    def test_in_target_compression(self):
        records = [
            _make_record(1000, 150),  # 15% - در محدوده
            _make_record(1000, 200),  # 20% - در محدوده
            _make_record(1000, 500),  # 50% - خارج
        ]
        result = analyze_split(records, "train")
        cr = result["compression_ratio"]
        assert cr["in_target_10_30_percent"] == pytest.approx(66.7, rel=0.01)


# ══════════════════════════════════════════════════════════
# تست load_jsonl
# ══════════════════════════════════════════════════════════


class TestLoadJsonl:

    def test_loads_valid_file(self, tmp_path):
        p = tmp_path / "test.jsonl"
        records = [{"id": i, "text": f"نص {i}"} for i in range(5)]
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        loaded = load_jsonl(p)
        assert len(loaded) == 5

    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_skips_empty_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"id": 1}\n\n{"id": 2}\n')
        loaded = load_jsonl(p)
        assert len(loaded) == 2

    def test_arabic_content_preserved(self, tmp_path):
        p = tmp_path / "test.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "النص العربي"}, ensure_ascii=False) + "\n")
        loaded = load_jsonl(p)
        assert loaded[0]["text"] == "النص العربي"


# ══════════════════════════════════════════════════════════
# تست generate_markdown_report
# ══════════════════════════════════════════════════════════


class TestGenerateMarkdownReport:

    @pytest.fixture
    def sample_analysis(self, sample_records):
        split_stats = analyze_split(sample_records, "train")
        return {
            "generated_at": "2024-01-01 00:00 UTC",
            "project": "arasum-offline",
            "sources": {
                "test_source": {
                    "source_name": "test_source",
                    "total_records": 5,
                    "total_in_range": 3,
                    "total_in_range_percent": 60.0,
                    "splits": {"train": split_stats},
                }
            },
        }

    def test_creates_file(self, tmp_path, sample_analysis):
        out = tmp_path / "report.md"
        generate_markdown_report(sample_analysis, out)
        assert out.exists()

    def test_contains_title(self, tmp_path, sample_analysis):
        out = tmp_path / "report.md"
        generate_markdown_report(sample_analysis, out)
        content = out.read_text(encoding="utf-8")
        assert "گزارش تحلیل دیتاست" in content

    def test_contains_source_name(self, tmp_path, sample_analysis):
        out = tmp_path / "report.md"
        generate_markdown_report(sample_analysis, out)
        content = out.read_text(encoding="utf-8")
        assert "test_source" in content

    def test_contains_statistics(self, tmp_path, sample_analysis):
        out = tmp_path / "report.md"
        generate_markdown_report(sample_analysis, out)
        content = out.read_text(encoding="utf-8")
        assert "میانگین" in content
        assert "میانه" in content
        

        