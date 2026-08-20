"""
محاسبه معیارهای ROUGE-1/ROUGE-2/ROUGE-L روی دیتاست تست.
"""

from rouge_score import rouge_scorer


def evaluate(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {key: sum(values) / len(values) for key, values in scores.items()}


if __name__ == "__main__":
    print("TODO: بارگذاری دیتاست تست و اجرای evaluate()")
