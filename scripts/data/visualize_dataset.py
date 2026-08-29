"""
تولید نمودارهای تحلیل بصری دیتاست آموزشی خلاصه‌سازی عربی.

خروجی‌ها (در reports/dataset_analysis/figures/):
    - 01_length_distribution.png     : توزیع طول متون و خلاصه‌ها
    - 02_compression_ratio.png       : توزیع نسبت فشرده‌سازی
    - 03_split_comparison.png        : مقایسه train/val/test
    - 04_length_scatter.png          : رابطه طول متن و خلاصه
    - 05_quality_dashboard.png       : داشبورد کیفیت کلی

اجرا:
    python scripts/data/visualize_dataset.py
    python scripts/data/visualize_dataset.py --source xlsum_arabic
    python scripts/data/visualize_dataset.py --dpi 150 --format pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# بررسی وجود matplotlib قبل از import
try:
    import matplotlib
    matplotlib.use("Agg")  # بدون نیاز به display - مناسب سرور
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════
# ثابت‌ها
# ══════════════════════════════════════════════════════════

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
REPORT_JSON = _PROJECT_ROOT / "reports" / "dataset_analysis" / "dataset_report.json"
FIGURES_DIR = _PROJECT_ROOT / "reports" / "dataset_analysis" / "figures"

# پالت رنگی حرفه‌ای - سازگار با چاپ سیاه‌وسفید
COLORS = {
    "train":      "#2563EB",   # آبی
    "validation": "#16A34A",   # سبز
    "test":       "#DC2626",   # قرمز
    "accent":     "#7C3AED",   # بنفش
    "neutral":    "#6B7280",   # خاکستری
    "warning":    "#D97706",   # نارنجی
    "bg":         "#F8FAFC",   # پس‌زمینه
    "grid":       "#E2E8F0",   # خطوط grid
}

SPLIT_ORDER = ["train", "validation", "test"]

# تنظیمات فونت - از فونت‌های پیش‌فرض استفاده می‌کند
FONT_CONFIG = {
    "family": "DejaVu Sans",
    "title_size": 14,
    "label_size": 11,
    "tick_size": 9,
    "annotation_size": 9,
}

# ══════════════════════════════════════════════════════════
# توابع کمکی
# ══════════════════════════════════════════════════════════


def _setup_style() -> None:
    """تنظیم استایل کلی نمودارها."""
    plt.rcParams.update({
        "figure.facecolor":       COLORS["bg"],
        "axes.facecolor":         "white",
        "axes.grid":              True,
        "grid.color":             COLORS["grid"],
        "grid.linewidth":         0.8,
        "grid.alpha":             0.7,
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.spines.left":       True,
        "axes.spines.bottom":     True,
        "axes.linewidth":         0.8,
        "font.family":            FONT_CONFIG["family"],
        "font.size":              FONT_CONFIG["tick_size"],
        "axes.titlesize":         FONT_CONFIG["title_size"],
        "axes.labelsize":         FONT_CONFIG["label_size"],
        "xtick.labelsize":        FONT_CONFIG["tick_size"],
        "ytick.labelsize":        FONT_CONFIG["tick_size"],
        "legend.fontsize":        FONT_CONFIG["tick_size"],
        "legend.framealpha":      0.9,
        "legend.edgecolor":       COLORS["grid"],
        "figure.dpi":             100,
        "savefig.bbox":           "tight",
        "savefig.facecolor":      COLORS["bg"],
    })


def _add_watermark(fig: "plt.Figure", text: str = "arasum-offline") -> None:
    """اضافه کردن watermark به نمودار."""
    fig.text(
        0.99, 0.01, text,
        ha="right", va="bottom",
        fontsize=7, color=COLORS["neutral"],
        alpha=0.5,
        transform=fig.transFigure,
    )


def _add_value_labels(
    ax: "plt.Axes",
    bars: Any,
    fmt: str = "{:.0f}",
    color: str = "white",
    fontsize: int = 8,
) -> None:
    """اضافه کردن label روی هر bar."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 0.5,
                fmt.format(h),
                ha="center", va="center",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
            )


def _load_report() -> dict[str, Any]:
    """بارگذاری گزارش JSON از analyze_dataset."""
    if not REPORT_JSON.exists():
        raise FileNotFoundError(
            f"گزارش JSON پیدا نشد: {REPORT_JSON}\n"
            "ابتدا این دستور را اجرا کنید:\n"
            "  python scripts/data/analyze_dataset.py"
        )
    with open(REPORT_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_raw_data(source_name: str) -> dict[str, list[dict]]:
    """
    بارگذاری داده خام از فایل‌های JSONL برای رسم scatter plot.
    نمونه‌برداری تصادفی برای عملکرد بهتر.
    """
    import random
    result: dict[str, list[dict]] = {}
    source_dir = PROCESSED_DIR / source_name

    for split in SPLIT_ORDER:
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path, encoding="utf-8") as f:
            all_lines = f.readlines()

        # نمونه‌برداری حداکثر ۲۰۰۰ رکورد
        sample_lines = random.sample(all_lines, min(2000, len(all_lines)))
        for line in sample_lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text_w = len(rec.get("text", "").split())
                summary_w = len(rec.get("summary", "").split())
                if text_w > 0 and summary_w > 0:
                    records.append({
                        "text_words": text_w,
                        "summary_words": summary_w,
                        "ratio": summary_w / text_w,
                    })
            except json.JSONDecodeError:
                pass
        result[split] = records
        logger.info("  %s [%s]: %d نمونه بارگذاری شد", source_name, split, len(records))

    return result


# ══════════════════════════════════════════════════════════
# نمودار ۱: توزیع طول متون
# ══════════════════════════════════════════════════════════

def plot_length_distribution(
    source_data: dict[str, Any],
    source_name: str,
    output_path: Path,
    dpi: int = 120,
) -> None:
    """
    نمودار ۱: توزیع طول متن ورودی و خلاصه مرجع.
    شامل histogram و box plot برای هر split.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Length Distribution - {source_name}",
        fontsize=FONT_CONFIG["title_size"] + 1,
        fontweight="bold",
        y=0.98,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    splits = source_data.get("splits", {})

    # ── ردیف اول: طول متن ورودی ──────────────────────────
    ax_main = fig.add_subplot(gs[0, :2])
    ax_box  = fig.add_subplot(gs[0, 2])

    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        tl = splits[split].get("text_length", {})
        if not tl:
            continue

        mean = tl["mean"]
        p25  = tl.get("p25", mean * 0.7)
        p75  = tl.get("p75", mean * 1.3)
        p10  = tl.get("p10", mean * 0.5)
        p90  = tl.get("p90", mean * 1.6)

        color = COLORS[split]

        ax_main.barh(
            split, mean,
            xerr=[[mean - p25], [p75 - mean]],
            color=color, alpha=0.75, height=0.5,
            capsize=5,
            error_kw={"linewidth": 2, "capthick": 2},
            label=f"{split} (mean={mean:.0f})",
        )
        # P10-P90 range به صورت شفاف
        ax_main.barh(
            split, p90 - p10, left=p10,
            color=color, alpha=0.15, height=0.5,
        )

    ax_main.axvline(300,  color=COLORS["warning"], linestyle=":", linewidth=1.5, label="Min (300)")
    ax_main.axvline(3000, color=COLORS["warning"], linestyle=":", linewidth=1.5, label="Max (3000)")
    ax_main.set_xlabel("Word Count")
    ax_main.set_title("Input Text Length  (Mean ± IQR  |  shading = P10-P90)", pad=8)
    ax_main.legend(fontsize=8, loc="lower right")
    ax_main.set_xlim(left=0)

    # ── box plot از آمار موجود ────────────────────────────
    bp_data:   list[list[float]] = []
    bp_labels: list[str]         = []

    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        tl = splits[split].get("text_length", {})
        if not tl:
            continue
        bp_data.append([
            float(tl.get("min",    tl["mean"] * 0.3)),
            float(tl.get("p25",    tl["mean"] * 0.7)),
            float(tl["median"]),
            float(tl.get("p75",    tl["mean"] * 1.3)),
            float(tl.get("max",    tl["mean"] * 2.0)),
        ])
        bp_labels.append(split)

    if bp_data and HAS_NUMPY:
        # ── سازگار با matplotlib ≥ 3.9 ──────────────────
        mpl_version = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
        boxplot_kwargs: dict[str, Any] = dict(
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            orientation="vertical",   # ← جایگزین vert=True
        )
        # در نسخه‌های قدیمی‌تر از 3.9: labels | جدیدتر: tick_labels
        if mpl_version >= (3, 9):
            boxplot_kwargs["tick_labels"] = bp_labels
        else:
            boxplot_kwargs["labels"] = bp_labels

        bplot = ax_box.boxplot(bp_data, **boxplot_kwargs)

        for patch, split in zip(bplot["boxes"], bp_labels):
            patch.set_facecolor(COLORS[split])
            patch.set_alpha(0.7)

    ax_box.axhline(300,  color=COLORS["warning"], linestyle=":", linewidth=1.2)
    ax_box.axhline(3000, color=COLORS["warning"], linestyle=":", linewidth=1.2)
    ax_box.set_title("Text Length Distribution", pad=8)
    ax_box.set_ylabel("Word Count")

    # ── ردیف دوم: طول خلاصه ──────────────────────────────
    ax_sum = fig.add_subplot(gs[1, :2])
    ax_cr  = fig.add_subplot(gs[1, 2])

    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        sl = splits[split].get("summary_length", {})
        if not sl:
            continue
        color = COLORS[split]
        mean  = sl["mean"]
        p25   = sl.get("p25", mean * 0.8)
        p75   = sl.get("p75", mean * 1.2)

        ax_sum.barh(
            split, mean,
            xerr=[[mean - p25], [p75 - mean]],
            color=color, alpha=0.75, height=0.5,
            capsize=5,
            error_kw={"linewidth": 2, "capthick": 2},
            label=f"{split} (mean={mean:.0f})",
        )

    ax_sum.set_xlabel("Word Count")
    ax_sum.set_title("Reference Summary Length  (Mean ± IQR)", pad=8)
    ax_sum.set_xlim(left=0)
    handles, labels = ax_sum.get_legend_handles_labels()
    if handles:
        ax_sum.legend(fontsize=8, loc="lower right")

    # ── compression ratio bar ─────────────────────────────
    cr_means:  list[float] = []
    cr_labels: list[str]   = []
    cr_colors: list[str]   = []

    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        cr = splits[split].get("compression_ratio", {})
        if not cr:
            continue
        cr_means.append(cr["mean"] * 100)
        cr_labels.append(split)
        cr_colors.append(COLORS[split])

    if cr_means:
        x_pos = list(range(len(cr_labels)))
        bars = ax_cr.bar(x_pos, cr_means, color=cr_colors, alpha=0.75, width=0.5)

        ax_cr.axhline(10, color=COLORS["warning"], linestyle="--",
                      linewidth=1.2, label="Target 10%")
        ax_cr.axhline(30, color=COLORS["warning"], linestyle="--",
                      linewidth=1.2, label="Target 30%")
        ax_cr.fill_between(
            [-0.5, len(cr_labels) - 0.5], 10, 30,
            alpha=0.08, color=COLORS["warning"],
        )
        for bar, val in zip(bars, cr_means):
            ax_cr.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                f"{val:.1f}%",
                ha="center", fontsize=9, fontweight="bold",
            )

        # ── fix warning: اول set_xticks بعد set_xticklabels ──
        ax_cr.set_xticks(x_pos)
        ax_cr.set_xticklabels(cr_labels)
        ax_cr.set_title("Compression Ratio", pad=8)
        ax_cr.set_ylabel("Summary / Text (%)")
        ax_cr.legend(fontsize=7)
        ax_cr.set_ylim(bottom=0)

    _add_watermark(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("نمودار ۱ ذخیره شد: %s", output_path)

# ══════════════════════════════════════════════════════════
# نمودار ۲: توزیع bin های طول
# ══════════════════════════════════════════════════════════


def plot_length_bins(
    source_data: dict[str, Any],
    source_name: str,
    output_path: Path,
    dpi: int = 120,
) -> None:
    """
    نمودار ۲: histogram دسته‌بندی شده طول متون.
    مقایسه توزیع در هر split.
    """
    splits = source_data.get("splits", {})
    available = [s for s in SPLIT_ORDER if s in splits]

    fig, axes = plt.subplots(
        1, len(available),
        figsize=(5 * len(available), 6),
        sharey=False,
    )
    if len(available) == 1:
        axes = [axes]

    fig.suptitle(
        f"Word Count Distribution by Bin - {source_name}",
        fontsize=FONT_CONFIG["title_size"] + 1,
        fontweight="bold",
        y=1.02,
    )

    BIN_LABELS_SHORT = [
        "<100", "100-300", "300-500", "500-1K",
        "1K-1.5K", "1.5K-2K", "2K-3K", ">3K",
    ]
    BIN_LABELS_FULL = [
        "خیلی کوتاه (<100)",
        "کوتاه (100-300)",
        "کوچک (300-500)",
        "متوسط (500-1000)",
        "متوسط‌بلند (1000-1500)",
        "بلند (1500-2000)",
        "خیلی بلند (2000-3000)",
        "خارج از محدوده (>3000)",
    ]

    for ax, split in zip(axes, available):
        dist = splits[split].get("length_distribution", {})
        total = splits[split].get("total_records", 1)

        values = [dist.get(label, 0) for label in BIN_LABELS_FULL]
        pcts   = [v / total * 100 for v in values]

        # رنگ‌بندی: در محدوده هدف = رنگ اصلی، خارج = خاکستری
        bar_colors = []
        for label in BIN_LABELS_FULL:
            if "300-500" in label or "500-1000" in label or \
               "1000-1500" in label or "1500-2000" in label or \
               "2000-3000" in label:
                bar_colors.append(COLORS[split])
            else:
                bar_colors.append(COLORS["neutral"])

        bars = ax.bar(
            range(len(BIN_LABELS_SHORT)),
            pcts,
            color=bar_colors,
            alpha=0.8,
            width=0.65,
            edgecolor="white",
            linewidth=0.5,
        )

        # label روی bar ها
        for bar, pct, val in zip(bars, pcts, values):
            if pct > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{pct:.0f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold",
                )

        ax.set_xticks(range(len(BIN_LABELS_SHORT)))
        ax.set_xticklabels(BIN_LABELS_SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(
            f"{split.capitalize()}\n({splits[split].get('total_records', 0):,} records)",
            pad=8, fontweight="bold",
        )

        max_pct = max(pcts) if pcts else 0
        ax.set_ylim(0, max(max_pct * 1.2, 1.0))

        # legend برای رنگ‌ها
        in_range = mpatches.Patch(color=COLORS[split], alpha=0.8, label="In range (300-3K)")
        out_range = mpatches.Patch(color=COLORS["neutral"], alpha=0.8, label="Out of range")
        ax.legend(handles=[in_range, out_range], fontsize=7, loc="upper right")

        # نمایش تعداد کل در محدوده
        in_range_pct = splits[split].get("in_range_percent", 0)
        ax.text(
            0.5, -0.22,
            f"In-range: {in_range_pct:.1f}%",
            transform=ax.transAxes,
            ha="center", fontsize=9,
            color=COLORS[split], fontweight="bold",
        )

    plt.tight_layout()
    _add_watermark(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("نمودار ۲ ذخیره شد: %s", output_path)


# ══════════════════════════════════════════════════════════
# نمودار ۳: مقایسه split ها
# ══════════════════════════════════════════════════════════


def plot_split_comparison(
    source_data: dict[str, Any],
    source_name: str,
    output_path: Path,
    dpi: int = 120,
) -> None:
    """
    نمودار ۳: مقایسه آماری train/validation/test در یک نگاه.
    """
    splits = source_data.get("splits", {})
    available = [s for s in SPLIT_ORDER if s in splits]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Split Comparison - {source_name}",
        fontsize=FONT_CONFIG["title_size"] + 1,
        fontweight="bold",
        y=0.98,
    )

    metrics = [
        ("total_records",     "Record Count",           "{:,.0f}", axes[0, 0]),
        ("in_range_percent",  "In-Range % (300-3000)",  "{:.1f}%", axes[0, 1]),
        ("has_summary_percent","Has Summary %",          "{:.1f}%", axes[1, 0]),
    ]

    split_colors = [COLORS[s] for s in available]

    for key, title, fmt, ax in metrics:
        values = [splits[s].get(key, 0) for s in available]
        bars = ax.bar(available, values, color=split_colors, alpha=0.8, width=0.5,
                      edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )
        ax.set_title(title, pad=8, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.2 if values else 10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )

    # نمودار ۴ام: text_length mean مقایسه با error bar
    ax_err = axes[1, 1]
    x_pos = range(len(available))
    for i, split in enumerate(available):
        tl = splits[split].get("text_length", {})
        mean = tl.get("mean", 0)
        std  = tl.get("std", 0)
        ax_err.bar(i, mean, color=COLORS[split], alpha=0.75, width=0.5)
        ax_err.errorbar(
            i, mean, yerr=std,
            fmt="none", color="black",
            capsize=6, capthick=2, linewidth=2,
        )
        ax_err.text(i, mean + std + 5, f"{mean:.0f}", ha="center",
                    fontsize=9, fontweight="bold")

    ax_err.axhline(300, color=COLORS["warning"], linestyle="--",
                   linewidth=1.2, label="Min 300")
    ax_err.axhline(3000, color=COLORS["warning"], linestyle="--",
                   linewidth=1.2, label="Max 3000")
    ax_err.set_xticks(list(x_pos))
    ax_err.set_xticklabels(available)
    ax_err.set_title("Mean Text Length ± Std", pad=8, fontweight="bold")
    ax_err.set_ylabel("Word Count")
    ax_err.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _add_watermark(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("نمودار ۳ ذخیره شد: %s", output_path)


# ══════════════════════════════════════════════════════════
# نمودار ۴: scatter plot طول متن vs خلاصه
# ══════════════════════════════════════════════════════════


def plot_length_scatter(
    raw_data: dict[str, list[dict]],
    source_name: str,
    output_path: Path,
    dpi: int = 120,
) -> None:
    """
    نمودار ۴: رابطه طول متن ورودی و خلاصه مرجع.
    هر نقطه یک نمونه از دیتاست است.
    """
    available = [s for s in SPLIT_ORDER if s in raw_data and raw_data[s]]

    fig, axes = plt.subplots(
        1, len(available),
        figsize=(5 * len(available), 5),
        sharey=True,
    )
    if len(available) == 1:
        axes = [axes]

    fig.suptitle(
        f"Text vs Summary Length Correlation - {source_name}",
        fontsize=FONT_CONFIG["title_size"] + 1,
        fontweight="bold",
        y=1.02,
    )

    for ax, split in zip(axes, available):
        records = raw_data[split]
        if not records:
            continue

        x = [r["text_words"] for r in records]
        y = [r["summary_words"] for r in records]
        ratios = [r["ratio"] for r in records]

        # رنگ‌بندی بر اساس نسبت فشرده‌سازی
        scatter = ax.scatter(
            x, y,
            c=ratios,
            cmap="RdYlGn",
            alpha=0.35,
            s=15,
            vmin=0.05, vmax=0.35,
        )

        # خط regression ساده
        if HAS_NUMPY and len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = sorted(x)
            ax.plot(x_line, [p(xi) for xi in x_line],
                    color=COLORS[split], linewidth=1.5,
                    linestyle="--", alpha=0.7, label=f"Trend")

        # محدوده هدف
        ax.axvline(300, color=COLORS["warning"], linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(3000, color=COLORS["warning"], linestyle=":", linewidth=1, alpha=0.7)

        # target compression band
        x_range = range(300, 3001, 100)
        ax.fill_between(
            x_range,
            [xi * 0.10 for xi in x_range],
            [xi * 0.30 for xi in x_range],
            alpha=0.08, color=COLORS["warning"],
            label="Target 10-30%",
        )

        ax.set_xlabel("Input Text (words)")
        if split == available[0]:
            ax.set_ylabel("Summary (words)")
        ax.set_title(
            f"{split.capitalize()} (n={len(records):,})",
            fontweight="bold", pad=8,
        )
        ax.legend(fontsize=7)

        # correlation coefficient
        if HAS_NUMPY and len(x) > 1:
            corr = float(np.corrcoef(x, y)[0, 1])
            ax.text(
                0.97, 0.05,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                ha="right", fontsize=9,
                color=COLORS[split], fontweight="bold",
            )

    # colorbar مشترک
    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                               norm=plt.Normalize(vmin=0.05, vmax=0.35))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Compression Ratio", fontsize=9)

    plt.tight_layout()
    _add_watermark(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("نمودار ۴ ذخیره شد: %s", output_path)


# ══════════════════════════════════════════════════════════
# نمودار ۵: داشبورد کیفیت کلی
# ══════════════════════════════════════════════════════════


def plot_quality_dashboard(
    source_data: dict[str, Any],
    source_name: str,
    output_path: Path,
    dpi: int = 120,
) -> None:
    """
    نمودار ۵: داشبورد کلی کیفیت دیتاست.
    مناسب‌ترین نمودار برای ارائه به ناظر فنی.
    """
    splits = source_data.get("splits", {})
    train  = splits.get("train", {})

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── عنوان اصلی ───────────────────────────────────────
    fig.text(
        0.5, 0.97,
        f"Dataset Quality Dashboard  |  {source_name}",
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        color="#1E293B",
    )
    fig.text(
        0.5, 0.935,
        "arasum-offline  |  Step 1: Data Preparation & Analysis",
        ha="center", va="top",
        fontsize=10, color=COLORS["neutral"],
    )

    # ── KPI کارت‌ها (ردیف اول) ───────────────────────────
    kpis = [
        ("Total Records", f"{source_data.get('total_records', 0):,}", COLORS["train"]),
        (
            "In-Range (300-3K)",
            f"{source_data.get('total_in_range_percent', 0):.1f}%",
            COLORS["validation"] if source_data.get("total_in_range_percent", 0) >= 50 else COLORS["warning"],
        ),
        (
            "Has Summary",
            f"{train.get('has_summary_percent', 0):.1f}%",
            COLORS["validation"] if train.get("has_summary_percent", 0) >= 90 else COLORS["warning"],
        ),
        (
            "Avg Text Length",
            f"{train.get('text_length', {}).get('mean', 0):.0f} w",
            COLORS["accent"],
        ),
    ]

    for i, (label, value, color) in enumerate(kpis):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.set_facecolor(color)
        ax_kpi.set_xlim(0, 1)
        ax_kpi.set_ylim(0, 1)
        ax_kpi.axis("off")
        ax_kpi.text(0.5, 0.62, value, ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white",
                    transform=ax_kpi.transAxes)
        ax_kpi.text(0.5, 0.25, label, ha="center", va="center",
                    fontsize=9, color="white", alpha=0.9,
                    transform=ax_kpi.transAxes)

    # ── نمودار میله‌ای: توزیع bin طول (ردیف دوم، چپ) ────
    ax_bins = fig.add_subplot(gs[1, :2])
    if train:
        dist  = train.get("length_distribution", {})
        total = train.get("total_records", 1)

        BIN_SHORT = ["<100","100-300","300-500","500-1K",
                     "1K-1.5K","1.5K-2K","2K-3K",">3K"]
        BIN_FULL  = [
            "خیلی کوتاه (<100)",
            "کوتاه (100-300)",
            "کوچک (300-500)",
            "متوسط (500-1000)",
            "متوسط‌بلند (1000-1500)",
            "بلند (1500-2000)",
            "خیلی بلند (2000-3000)",
            "خارج از محدوده (>3000)",
        ]
        IN_RANGE_BINS = {
            "کوچک (300-500)", "متوسط (500-1000)",
            "متوسط‌بلند (1000-1500)", "بلند (1500-2000)",
            "خیلی بلند (2000-3000)",
        }

        pcts   = [dist.get(bl, 0) / total * 100 for bl in BIN_FULL]
        colors = [COLORS["train"] if bl in IN_RANGE_BINS else COLORS["neutral"]
                  for bl in BIN_FULL]

        bars = ax_bins.bar(BIN_SHORT, pcts, color=colors, alpha=0.85,
                           width=0.65, edgecolor="white")
        for bar, pct in zip(bars, pcts):
            if pct > 0.5:
                ax_bins.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{pct:.1f}%",
                    ha="center", fontsize=7.5, fontweight="bold",
                )
        ax_bins.set_title("Train: Word Count Distribution", fontweight="bold", pad=6)
        ax_bins.set_xlabel("Length Bin")
        ax_bins.set_ylabel("Percentage (%)")
        ax_bins.set_xticks(range(len(BIN_SHORT)))
        ax_bins.set_xticklabels(BIN_SHORT, rotation=30, ha="right", fontsize=8)

        in_range_patch = mpatches.Patch(color=COLORS["train"], alpha=0.85,
                                         label="In target range (300-3K)")
        out_patch = mpatches.Patch(color=COLORS["neutral"], alpha=0.85,
                                    label="Out of range")
        ax_bins.legend(handles=[in_range_patch, out_patch], fontsize=7)

    # ── نمودار مقایسه نسبت فشرده‌سازی (ردیف دوم، راست) ──
    ax_cr = fig.add_subplot(gs[1, 2:])
    available = [s for s in SPLIT_ORDER if s in splits]
    cr_data = {
        split: splits[split].get("compression_ratio", {})
        for split in available
    }

    x_pos = range(len(available))
    means  = [cr_data[s].get("mean", 0) * 100  for s in available]
    p25s   = [cr_data[s].get("p25",  0) * 100  for s in available]
    p75s   = [cr_data[s].get("p75",  0) * 100  for s in available]

    bars = ax_cr.bar(list(x_pos), means,
                     color=[COLORS[s] for s in available],
                     alpha=0.8, width=0.5)
    for i, (s, mean, p25, p75) in enumerate(zip(available, means, p25s, p75s)):
        ax_cr.errorbar(i, mean,
                       yerr=[[mean - p25], [p75 - mean]],
                       fmt="none", color="black",
                       capsize=5, linewidth=1.5)
        ax_cr.text(i, p75 + 0.3, f"{mean:.1f}%",
                   ha="center", fontsize=9, fontweight="bold")

    ax_cr.axhline(10, color=COLORS["warning"], linestyle="--",
                  linewidth=1.5, label="Target min 10%")
    ax_cr.axhline(30, color=COLORS["warning"], linestyle="--",
                  linewidth=1.5, label="Target max 30%")
    ax_cr.fill_between([-0.5, len(available) - 0.5], 10, 30,
                       alpha=0.08, color=COLORS["warning"])
    ax_cr.set_xticks(list(x_pos))
    ax_cr.set_xticklabels([s.capitalize() for s in available])
    ax_cr.set_title("Compression Ratio by Split", fontweight="bold", pad=6)
    ax_cr.set_ylabel("Summary / Text (%)")
    ax_cr.legend(fontsize=8)
    ax_cr.set_ylim(bottom=0)

    # ── جدول آمار (ردیف سوم) ─────────────────────────────
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    col_labels = ["Split", "Records", "In-Range%",
                  "Has Summary%", "Text Mean(w)",
                  "Summary Mean(w)", "Compression%"]
    table_data = []
    for split in available:
        s = splits[split]
        tl = s.get("text_length", {})
        sl = s.get("summary_length", {})
        cr = s.get("compression_ratio", {})
        table_data.append([
            split.capitalize(),
            f"{s.get('total_records', 0):,}",
            f"{s.get('in_range_percent', 0):.1f}%",
            f"{s.get('has_summary_percent', 0):.1f}%",
            f"{tl.get('mean', 0):.0f}",
            f"{sl.get('mean', 0):.0f}",
            f"{cr.get('mean', 0):.1%}",
        ])

    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # استایل header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#1E293B")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # استایل ردیف‌ها
        for i, split in enumerate(available):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_facecolor(
                    matplotlib.colors.to_rgba(COLORS[split], 0.12)
                )

    _add_watermark(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info("نمودار ۵ ذخیره شد: %s", output_path)


# ══════════════════════════════════════════════════════════
# CLI و main
# ══════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="تولید نمودارهای تحلیل بصری دیتاست",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--source", type=str, default=None,
                        help="نام سورس مشخص (پیش‌فرض: همه)")
    parser.add_argument("--dpi", type=int, default=120,
                        help="کیفیت تصویر (پیش‌فرض: 120)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="فرمت خروجی (پیش‌فرض: png)")
    parser.add_argument("--skip-scatter", action="store_true",
                        help="رد کردن scatter plot (کندتر است)")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR,
                        help=f"پوشه خروجی (پیش‌فرض: {FIGURES_DIR})")
    return parser.parse_args()


def main() -> None:
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib نصب نیست. دستور نصب:")
        print("   pip install matplotlib")
        sys.exit(1)

    if not HAS_NUMPY:
        print("⚠️  numpy نصب نیست - برخی ویژگی‌ها غیرفعال می‌شوند.")

    args = parse_args()
    _setup_style()

    # بارگذاری گزارش
    report = _load_report()
    sources = report.get("sources", {})

    if args.source:
        if args.source not in sources:
            logger.error("سورس '%s' در گزارش پیدا نشد.", args.source)
            sys.exit(1)
        sources = {args.source: sources[args.source]}

    output_dir: Path = args.output_dir
    fmt = args.format

    print(f"\n📊 تولید نمودارها برای {len(sources)} سورس...\n")

    for src_name, src_data in sources.items():
        print(f"  📁 {src_name}")

        plot_length_distribution(
            src_data, src_name,
            output_dir / f"01_length_distribution.{fmt}",
            dpi=args.dpi,
        )
        print(f"    ✅ 01_length_distribution.{fmt}")

        plot_length_bins(
            src_data, src_name,
            output_dir / f"02_length_bins.{fmt}",
            dpi=args.dpi,
        )
        print(f"    ✅ 02_length_bins.{fmt}")

        plot_split_comparison(
            src_data, src_name,
            output_dir / f"03_split_comparison.{fmt}",
            dpi=args.dpi,
        )
        print(f"    ✅ 03_split_comparison.{fmt}")

        if not args.skip_scatter:
            raw_data = _load_raw_data(src_name)
            if any(raw_data.values()):
                plot_length_scatter(
                    raw_data, src_name,
                    output_dir / f"04_length_scatter.{fmt}",
                    dpi=args.dpi,
                )
                print(f"    ✅ 04_length_scatter.{fmt}")

        plot_quality_dashboard(
            src_data, src_name,
            output_dir / f"05_quality_dashboard.{fmt}",
            dpi=args.dpi,
        )
        print(f"    ✅ 05_quality_dashboard.{fmt}")

    print(f"\n✅ همه نمودارها در: {output_dir}\n")


if __name__ == "__main__":
    main()