from arabic_summarizer.postprocessing.formatter import format_summary


def test_format_summary_adds_final_dot():
    result = format_summary("این یک خلاصه است")
    assert result.endswith(".")


def test_format_summary_removes_extra_spaces():
    result = format_summary("این    یک   خلاصه است.")
    assert "  " not in result
