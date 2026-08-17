"""
نرمال‌سازی حروف عربی (مثل تبدیل أ/إ/آ به ا).
TODO: می‌توان از camel_tools.utils.normalize استفاده کرد.
"""

NORMALIZATION_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ى": "ي", "ة": "ه",
    "ؤ": "و", "ئ": "ي",
}


def normalize_text(text: str) -> str:
    """اعمال نگاشت نرمال‌سازی حروف عربی"""
    for src, target in NORMALIZATION_MAP.items():
        text = text.replace(src, target)
    return text
