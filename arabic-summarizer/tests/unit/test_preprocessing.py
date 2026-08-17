"""
تست‌های واحد برای ماژول‌های پیش‌پردازش.

اجرا:
    pytest tests/unit/test_preprocessing.py -v
    pytest tests/unit/test_preprocessing.py -v --cov=src/arabic_summarizer/preprocessing
"""

from __future__ import annotations

import pytest

from arabic_summarizer.exceptions import InvalidInputError, PreprocessingError
from arabic_summarizer.preprocessing import ArabicPreprocessingPipeline, PreprocessingResult
from arabic_summarizer.preprocessing.cleaner import ArabicTextCleaner
from arabic_summarizer.preprocessing.normalizer import ArabicNormalizer


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def normalizer() -> ArabicNormalizer:
    return ArabicNormalizer()


@pytest.fixture
def cleaner() -> ArabicTextCleaner:
    return ArabicTextCleaner(min_words=300, max_words=3000)


@pytest.fixture
def pipeline() -> ArabicPreprocessingPipeline:
    # CamelTools را غیرفعال می‌کنیم تا تست محیط مستقل داشته باشد
    return ArabicPreprocessingPipeline(
        min_words=300,
        max_words=3000,
        normalize_teh=False,
        use_camel_tools=False,
    )


def _make_arabic_text(word_count: int) -> str:
    """تولید متن عربی ساختگی با تعداد کلمات مشخص."""
    word = "كلمة"
    return " ".join([word] * word_count)


# ═══════════════════════════════════════════════════════════
# تست‌های ArabicNormalizer
# ═══════════════════════════════════════════════════════════


class TestArabicNormalizer:

    def test_normalize_alef_hamza_above(self, normalizer):
        """أ باید به ا تبدیل شود."""
        assert normalizer.normalize_alef("أحمد") == "احمد"

    def test_normalize_alef_hamza_below(self, normalizer):
        """إ باید به ا تبدیل شود."""
        assert normalizer.normalize_alef("إبراهيم") == "ابراهيم"

    def test_normalize_alef_madda(self, normalizer):
        """آ باید به ا تبدیل شود."""
        assert normalizer.normalize_alef("آدم") == "ادم"

    def test_normalize_alef_wasla(self, normalizer):
        """ٱ باید به ا تبدیل شود."""
        assert normalizer.normalize_alef("ٱلله") == "الله"

    def test_normalize_alef_no_change_for_plain(self, normalizer):
        """الف ساده نباید تغییر کند."""
        assert normalizer.normalize_alef("الكتاب") == "الكتاب"

    def test_normalize_yeh(self, normalizer):
        """الف مقصوره باید به یاء تبدیل شود."""
        assert normalizer.normalize_yeh("موسى") == "موسي"

    def test_remove_diacritics_fatha(self, normalizer):
        """فتحه باید حذف شود."""
        assert normalizer.remove_diacritics("كَتَبَ") == "كتب"

    def test_remove_diacritics_kasra(self, normalizer):
        """کسره باید حذف شود."""
        assert normalizer.remove_diacritics("بِسْمِ") == "بسم"

    def test_remove_diacritics_full_sentence(self, normalizer):
        """جمله با اعراب کامل باید بدون اعراب شود."""
        result = normalizer.remove_diacritics("الْحَمْدُ لِلَّهِ")
        assert result == "الحمد لله"

    def test_remove_tatweel(self, normalizer):
        """کشیده باید حذف شود."""
        assert normalizer.remove_tatweel("كتاااب") == "كتاب"

    def test_normalize_whitespace_multiple_spaces(self, normalizer):
        """چند فاصله باید به یک فاصله تبدیل شود."""
        assert normalizer.normalize_whitespace("كلمة   أخرى") == "كلمة أخرى"

    def test_normalize_whitespace_strips(self, normalizer):
        """فاصله ابتدا و انتها باید حذف شود."""
        assert normalizer.normalize_whitespace("  كلمة  ") == "كلمة"

    def test_normalize_full_pipeline_order(self, normalizer):
        """متد normalize باید ترتیب صحیح را رعایت کند."""
        result = normalizer.normalize("مَدْرَسَةٌ أحمد")
        # اعراب حذف شده، الف نرمال شده
        assert "َ" not in result
        assert "ً" not in result
        assert "أ" not in result

    def test_normalize_plain_text_no_change(self, normalizer):
        """متن بدون نیاز به نرمال‌سازی نباید تغییر کند."""
        text = "الكتاب والقلم"
        result = normalizer.normalize(text)
        assert result == text

    @pytest.mark.parametrize(
        "input_text,expected_fragment",
        [
            ("إبراهيم", "ابراهيم"),
            ("أُسْتَاذٌ", "استاذ"),
            ("آيَة", "اية"),
        ],
    )
    def test_normalize_parametric(self, normalizer, input_text, expected_fragment):
        """تست parametric برای موارد مختلف نرمال‌سازی."""
        result = normalizer.normalize(input_text)
        assert expected_fragment in result


# ═══════════════════════════════════════════════════════════
# تست‌های ArabicTextCleaner
# ═══════════════════════════════════════════════════════════


class TestArabicTextCleaner:

    def test_remove_http_url(self, cleaner):
        """URL با http باید حذف شود."""
        text = "اقرأ المزيد على https://example.com اليوم"
        result = cleaner.remove_urls(text)
        assert "https://example.com" not in result
        assert "اقرأ" in result

    def test_remove_www_url(self, cleaner):
        """URL با www باید حذف شود."""
        text = "زيارة www.example.com للمزيد"
        result = cleaner.remove_urls(text)
        assert "www.example.com" not in result

    def test_remove_html_tags(self, cleaner):
        """تگ‌های HTML باید حذف شوند."""
        text = "<p>النص <b>العربي</b> هنا</p>"
        result = cleaner.remove_html_tags(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "النص" in result
        assert "العربي" in result

    def test_remove_html_keeps_content(self, cleaner):
        """محتوای داخل تگ‌ها باید حفظ شود."""
        result = cleaner.remove_html_tags("<div>محتوى</div>")
        assert "محتوى" in result

    def test_normalize_repeated_exclamation(self, cleaner):
        """!!! باید به ! تبدیل شود."""
        result = cleaner.normalize_punctuation("رائع!!!")
        assert result == "رائع!"

    def test_normalize_repeated_question(self, cleaner):
        """؟؟؟ باید به ؟ تبدیل شود."""
        result = cleaner.normalize_punctuation("ماذا؟؟؟")
        assert result == "ماذا؟"

    def test_count_words_basic(self, cleaner):
        """شمارش کلمات باید درست باشد."""
        assert cleaner.count_words("كلمة أخرى ثالثة") == 3

    def test_validate_length_below_min_raises(self, cleaner):
        """متن با ۲۹۹ کلمه باید InvalidInputError بدهد."""
        short_text = _make_arabic_text(299)
        with pytest.raises(InvalidInputError) as exc_info:
            cleaner.validate_length(short_text)
        assert exc_info.value.word_count == 299

    def test_validate_length_above_max_raises(self, cleaner):
        """متن با ۳۰۰۱ کلمه باید InvalidInputError بدهد."""
        long_text = _make_arabic_text(3001)
        with pytest.raises(InvalidInputError) as exc_info:
            cleaner.validate_length(long_text)
        assert exc_info.value.word_count == 3001

    def test_validate_length_at_min_passes(self, cleaner):
        """متن با دقیقاً ۳۰۰ کلمه باید قبول شود."""
        text = _make_arabic_text(300)
        result = cleaner.validate_length(text)
        assert result == 300

    def test_validate_length_at_max_passes(self, cleaner):
        """متن با دقیقاً ۳۰۰۰ کلمه باید قبول شود."""
        text = _make_arabic_text(3000)
        result = cleaner.validate_length(text)
        assert result == 3000

    def test_validate_length_in_range_passes(self, cleaner):
        """متن ۵۰۰ کلمه‌ای باید بدون خطا باشد."""
        text = _make_arabic_text(500)
        result = cleaner.validate_length(text)
        assert result == 500

    def test_clean_and_validate_returns_tuple(self, cleaner):
        """clean_and_validate باید tuple برگرداند."""
        text = _make_arabic_text(400)
        result = cleaner.clean_and_validate(text)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cleaned_text, word_count = result
        assert isinstance(cleaned_text, str)
        assert isinstance(word_count, int)

    def test_clean_removes_url_before_validation(self, cleaner):
        """URL باید قبل از شمارش کلمات حذف شود."""
        # اگر URL در شمارش کلمات حساب شود ممکن است طول تغییر کند
        arabic_words = _make_arabic_text(400)
        text_with_url = arabic_words + " https://example.com"
        # نباید خطا بدهد چون URL حذف می‌شود
        cleaned, count = cleaner.clean_and_validate(text_with_url)
        assert "https://example.com" not in cleaned


# ═══════════════════════════════════════════════════════════
# تست‌های ArabicPreprocessingPipeline
# ═══════════════════════════════════════════════════════════


class TestArabicPreprocessingPipeline:

    def test_pipeline_returns_result_object(self, pipeline):
        """pipeline باید PreprocessingResult برگرداند."""
        text = _make_arabic_text(400)
        result = pipeline.run(text)
        assert isinstance(result, PreprocessingResult)

    def test_pipeline_result_has_cleaned_text(self, pipeline):
        """نتیجه باید cleaned_text داشته باشد."""
        text = _make_arabic_text(400)
        result = pipeline.run(text)
        assert isinstance(result.cleaned_text, str)
        assert len(result.cleaned_text) > 0

    def test_pipeline_result_has_word_count(self, pipeline):
        """نتیجه باید word_count داشته باشد."""
        text = _make_arabic_text(400)
        result = pipeline.run(text)
        assert result.word_count > 0

    def test_pipeline_camel_tools_flag(self, pipeline):
        """چون CamelTools غیرفعال است، camel_tools_used باید False باشد."""
        text = _make_arabic_text(400)
        result = pipeline.run(text)
        assert result.camel_tools_used is False

    def test_pipeline_invalid_short_text_raises(self, pipeline):
        """متن کوتاه‌تر از ۳۰۰ کلمه باید InvalidInputError بدهد."""
        short_text = _make_arabic_text(100)
        with pytest.raises(InvalidInputError):
            pipeline.run(short_text)

    def test_pipeline_invalid_long_text_raises(self, pipeline):
        """متن بلندتر از ۳۰۰۰ کلمه باید InvalidInputError بدهد."""
        long_text = _make_arabic_text(3500)
        with pytest.raises(InvalidInputError):
            pipeline.run(long_text)

    def test_pipeline_removes_diacritics(self, pipeline):
        """pipeline باید اعراب را حذف کند."""
        diacritized_words = " ".join(["كَتَبَ"] * 400)
        result = pipeline.run(diacritized_words)
        assert "َ" not in result.cleaned_text

    def test_pipeline_normalizes_alef(self, pipeline):
        """pipeline باید الف را نرمال کند."""
        text_with_alef = " ".join(["أحمد"] * 400)
        result = pipeline.run(text_with_alef)
        assert "أ" not in result.cleaned_text

    def test_pipeline_removes_urls(self, pipeline):
        """pipeline باید URL را حذف کند."""
        arabic_words = _make_arabic_text(398)
        text = arabic_words + " https://example.com كلمة"
        result = pipeline.run(text)
        assert "https://example.com" not in result.cleaned_text

    def test_pipeline_processes_real_arabic_text(self, pipeline):
        """
        تست با متن واقعی عربی.
        یک پاراگراف خبری واقعی با تعداد کافی کلمات.
        """
        real_text = (
            "أعلنت وزارة التعليم عن إطلاق برنامج جديد لتطوير المناهج الدراسية "
            "في المرحلة الثانوية، ويهدف البرنامج إلى تحديث أساليب التدريس "
            "وتعزيز مهارات التفكير النقدي لدى الطلاب. "
        ) * 30  # تکرار برای رسیدن به ۳۰۰ کلمه

        result = pipeline.run(real_text)
        assert isinstance(result.cleaned_text, str)
        assert result.word_count >= 300
        assert len(result.cleaned_text) > 0