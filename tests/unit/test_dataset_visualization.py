"""
Unit Tests برای visualize_dataset.
بدون نیاز به display یا فایل واقعی اجرا می‌شوند.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "data"))

matplotlib = pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

from visualize_dataset import (
    COLORS,
    SPLIT_ORDER,
    _setup_style,
    plot_length_distribution,
    plot_length_bins,
    plot_split_comparison,
    plot_quality_dashboard,
)


# ══════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════


def _make_split_stats(
    n: int = 1000,
    mean_text: float = 600.0,
    mean_summary: float = 30.0,
) -> dict:
    return {
        "total_records": n,
        "has_summary": n,
        "has_summary_percent": 100.0,
        "in_range_300_3000": int(n * 0.85),
        "in_range_percent": 85.0,
        "low_arabic_ratio_count": 5,
        "text_length": {
            "mean": mean_text,
            "median": mean_text * 0.9,
            "std": mean_text * 0.3,
            "min": 300,
            "max": 2500,
            "p10": mean_text * 0.5,
            "p25": mean_text * 0.7,
            "p75": mean_text * 1.3,
            "p90": mean_text * 1.6,
        },
        "summary_length": {
            "mean": mean_summary,
            "median": mean_summary * 0.9,
            "std": mean_summary * 0.25,
            "min": 10,
            "max": 80,
            "p25": mean_summary * 0.8,
            "p75": mean_summary * 1.2,
        },
        "compression_ratio": {
            "mean": mean_summary / mean_text,
            "median": (mean_summary * 0.9) / mean_text,
            "min": 0.02,
            "max": 0.35,
            "p25": 0.03,
            "p75": 0.08,
            "in_target_10_30_percent": 42.0,
        },
        "length_distribution": {
            "خیلی کوتاه (<100)":        0,
            "کوتاه (100-300)":           0,
            "کوچک (300-500)":            350,
            "متوسط (500-1000)":          430,
            "متوسط‌بلند (1000-1500)":    150,
            "بلند (1500-2000)":          50,
            "خیلی بلند (2000-3000)":     20,
            "خارج از محدوده (>3000)":    0,
        },
        "arabic_ratio": {"mean": 0.92, "min": 0.75},
    }


@pytest.fixture
def sample_source_data():
    return {
        "source_name": "test_source",
        "total_records": 3000,
        "total_in_range": 2550,
        "total_in_range_percent": 85.0,
        "splits": {
            "train":      _make_split_stats(2400, 600.0, 30.0),
            "validation": _make_split_stats(300,  615.0, 31.0),
            "test":       _make_split_stats(300,  610.0, 29.0),
        },
    }


@pytest.fixture
def sample_raw_data():
    import random
    result = {}
    for split in ["train", "validation", "test"]:
        n = 200 if split == "train" else 50
        result[split] = [
            {
                "text_words": random.randint(300, 2000),
                "summary_words": random.randint(20, 80),
                "ratio": random.uniform(0.03, 0.15),
            }
            for _ in range(n)
        ]
    return result


# ══════════════════════════════════════════════════════════
# تست setup
# ══════════════════════════════════════════════════════════


class TestSetup:

    def test_setup_style_runs(self):
        """setup_style باید بدون خطا اجرا شود."""
        _setup_style()

    def test_colors_have_required_keys(self):
        required = {"train", "validation", "test", "warning", "neutral"}
        assert required.issubset(COLORS.keys())

    def test_split_order_correct(self):
        assert SPLIT_ORDER == ["train", "validation", "test"]


# ══════════════════════════════════════════════════════════
# تست plot_length_distribution
# ══════════════════════════════════════════════════════════


class TestPlotLengthDistribution:

    def test_creates_file(self, tmp_path, sample_source_data):
        out = tmp_path / "01.png"
        plot_length_distribution(sample_source_data, "test", out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_works_with_single_split(self, tmp_path):
        data = {
            "source_name": "s",
            "total_records": 100,
            "total_in_range": 80,
            "total_in_range_percent": 80.0,
            "splits": {"train": _make_split_stats(100)},
        }
        out = tmp_path / "01_single.png"
        plot_length_distribution(data, "s", out, dpi=72)
        assert out.exists()

    def test_works_with_no_summary(self, tmp_path):
        """وقتی خلاصه‌ای وجود ندارد باید بدون crash اجرا شود."""
        stats = _make_split_stats(100)
        stats.pop("summary_length")
        stats.pop("compression_ratio")
        stats["has_summary"] = 0
        stats["has_summary_percent"] = 0.0
        data = {
            "source_name": "s",
            "total_records": 100,
            "total_in_range": 80,
            "total_in_range_percent": 80.0,
            "splits": {"train": stats},
        }
        out = tmp_path / "01_no_summary.png"
        plot_length_distribution(data, "s", out, dpi=72)
        assert out.exists()


# ══════════════════════════════════════════════════════════
# تست plot_length_bins
# ══════════════════════════════════════════════════════════


class TestPlotLengthBins:

    def test_creates_file(self, tmp_path, sample_source_data):
        out = tmp_path / "02.png"
        plot_length_bins(sample_source_data, "test", out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_all_zero_distribution(self, tmp_path):
        """توزیع خالی نباید crash کند."""
        stats = _make_split_stats(100)
        stats["length_distribution"] = {}
        data = {
            "source_name": "s",
            "total_records": 100,
            "total_in_range": 0,
            "total_in_range_percent": 0.0,
            "splits": {"train": stats},
        }
        out = tmp_path / "02_empty.png"
        plot_length_bins(data, "s", out, dpi=72)
        assert out.exists()


# ══════════════════════════════════════════════════════════
# تست plot_split_comparison
# ══════════════════════════════════════════════════════════


class TestPlotSplitComparison:

    def test_creates_file(self, tmp_path, sample_source_data):
        out = tmp_path / "03.png"
        plot_split_comparison(sample_source_data, "test", out, dpi=72)
        assert out.exists()

    def test_single_split_no_crash(self, tmp_path):
        data = {
            "source_name": "s",
            "total_records": 500,
            "total_in_range": 400,
            "total_in_range_percent": 80.0,
            "splits": {"train": _make_split_stats(500)},
        }
        out = tmp_path / "03_single.png"
        plot_split_comparison(data, "s", out, dpi=72)
        assert out.exists()


# ══════════════════════════════════════════════════════════
# تست plot_quality_dashboard
# ══════════════════════════════════════════════════════════


class TestPlotQualityDashboard:

    def test_creates_file(self, tmp_path, sample_source_data):
        out = tmp_path / "05.png"
        plot_quality_dashboard(sample_source_data, "test", out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 5000

    def test_file_is_valid_image(self, tmp_path, sample_source_data):
        """بررسی اینکه فایل PNG معتبر است."""
        out = tmp_path / "05_valid.png"
        plot_quality_dashboard(sample_source_data, "test", out, dpi=72)
        with open(out, "rb") as f:
            header = f.read(8)
        # PNG signature
        assert header[:4] == b"\x89PNG"

    def test_empty_splits_no_crash(self, tmp_path):
        data = {
            "source_name": "empty",
            "total_records": 0,
            "total_in_range": 0,
            "total_in_range_percent": 0.0,
            "splits": {},
        }
        out = tmp_path / "05_empty.png"
        plot_quality_dashboard(data, "empty", out, dpi=72)
        assert out.exists()