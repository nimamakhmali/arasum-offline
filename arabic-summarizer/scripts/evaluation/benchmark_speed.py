"""
بنچ‌مارک سرعت اجرا روی CPU/GPU - خروجی CSV برای گزارش نهایی.
ستون‌های خروجی: text_size, latency, memory, cpu_usage
"""

import csv
import time


def run_benchmark(summarizer, texts: list, output_csv: str = "benchmark_results.csv"):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_size_words", "latency_seconds"])

        for text in texts:
            start = time.time()
            summarizer.summarize(text)
            elapsed = time.time() - start
            writer.writerow([len(text.split()), round(elapsed, 3)])


if __name__ == "__main__":
    print("TODO: بارگذاری نمونه متون و اجرای run_benchmark()")
