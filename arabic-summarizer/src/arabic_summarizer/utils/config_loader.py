"""
بارگذاری و مدیریت تنظیمات پروژه از فایل‌های YAML.
از الگوی Singleton استفاده می‌کند تا فایل‌ها فقط یک‌بار خوانده شوند.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from arabic_summarizer.utils.logger import get_logger

logger = get_logger(__name__)

# مسیر پیش‌فرض پوشه configs نسبت به ریشه پروژه
_DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[4] / "configs"


class ConfigLoader:
    """
    بارگذاری تنظیمات از فایل‌های YAML.

    استفاده:
        config = ConfigLoader()
        model_name = config.get("model_config", "model.name")
    """

    _instance: ConfigLoader | None = None
    _cache: dict[str, dict[str, Any]] = {}

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        configs_dir_env = os.environ.get("CONFIGS_DIR", "")
        self._configs_dir = (
            Path(configs_dir_env) if configs_dir_env else _DEFAULT_CONFIGS_DIR
        )

    def _load_file(self, config_name: str) -> dict[str, Any]:
        """
        فایل YAML را می‌خواند و در کش ذخیره می‌کند.
        اگر فایل وجود نداشت دیکشنری خالی برمی‌گرداند.
        """
        if config_name in self._cache:
            return self._cache[config_name]

        file_path = self._configs_dir / f"{config_name}.yaml"

        if not file_path.exists():
            logger.warning(
                "فایل تنظیمات پیدا نشد، از مقادیر پیش‌فرض استفاده می‌شود: %s",
                file_path,
            )
            self._cache[config_name] = {}
            return {}

        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[config_name] = data
        logger.debug("تنظیمات بارگذاری شد: %s", file_path.name)
        return data

    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        یک مقدار از فایل تنظیمات می‌خواند.

        Args:
            config_name: نام فایل بدون پسوند (مثلاً "model_config")
            key_path: مسیر کلید با نقطه (مثلاً "model.name")
            default: مقدار پیش‌فرض اگر کلید وجود نداشت

        Returns:
            مقدار تنظیم یا default
        """
        data = self._load_file(config_name)
        keys = key_path.split(".")
        value = data

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]

        return value

    def get_section(self, config_name: str, section: str) -> dict[str, Any]:
        """
        یک بخش کامل از فایل تنظیمات برمی‌گرداند.

        Args:
            config_name: نام فایل بدون پسوند
            section: نام بخش اصلی (مثلاً "model")

        Returns:
            دیکشنری بخش یا دیکشنری خالی
        """
        data = self._load_file(config_name)
        return data.get(section, {})

    def reload(self) -> None:
        """کش تنظیمات را پاک می‌کند تا فایل‌ها مجدداً خوانده شوند."""
        self._cache.clear()
        logger.info("کش تنظیمات پاک شد")


# نمونه سراسری برای استفاده آسان
config = ConfigLoader()