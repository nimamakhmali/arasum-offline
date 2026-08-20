"""
ماژول لاگ‌گیری مرکزی پروژه.
تمام ماژول‌های دیگر از این ماژول logger می‌گیرند.
"""

import logging
import os
import sys
from pathlib import Path


LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _initialize_logging() -> None:
    """
    لاگ‌گیری را یک‌بار در طول عمر برنامه راه‌اندازی می‌کند.
    سطح لاگ از متغیر محیطی LOG_LEVEL خوانده می‌شود.
    پیش‌فرض: INFO
    """
    global _initialized
    if _initialized:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = LOG_LEVEL_MAP.get(level_name, logging.INFO)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Handler کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Handler فایل (اختیاری - فعال اگر LOG_FILE تنظیم شده باشد)
    handlers: list[logging.Handler] = [console_handler]

    log_file = os.environ.get("LOG_FILE", "")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    یک logger با نام مشخص برمی‌گرداند.

    Args:
        name: نام ماژول، معمولاً __name__ پاس داده می‌شود.

    Returns:
        Logger آماده استفاده.

    Example:
        logger = get_logger(__name__)
        logger.info("سیستم راه‌اندازی شد")
    """
    _initialize_logging()
    return logging.getLogger(name)