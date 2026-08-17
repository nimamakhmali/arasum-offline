from arabic_summarizer.preprocessing.cleaner import clean_text
from arabic_summarizer.preprocessing.normalizer import normalize_text


def test_clean_text_removes_extra_spaces():
    assert clean_text("سلام   دنیا") == "سلام دنیا"


def test_normalize_text_converts_alef_variants():
    assert normalize_text("أحمد إبراهيم آدم") == "احمد ابراهيم ادم"
