from arabic_summarizer.utils.hashing import generate_cache_key


def test_same_text_and_ratio_produce_same_key():
    key1 = generate_cache_key("متن نمونه", 0.2)
    key2 = generate_cache_key("متن نمونه", 0.2)
    assert key1 == key2


def test_different_ratio_produces_different_key():
    key1 = generate_cache_key("متن نمونه", 0.2)
    key2 = generate_cache_key("متن نمونه", 0.3)
    assert key1 != key2
