"""
تبدیل Exception های اختصاصی پروژه به پاسخ‌های مناسب HTTP.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from arabic_summarizer.exceptions import (
    ModelLoadError, InferenceError, ValidationError, CacheError
)


async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": str(exc)})


async def inference_error_handler(request: Request, exc: InferenceError):
    return JSONResponse(status_code=500, content={"error": "inference_error", "detail": str(exc)})


async def model_load_error_handler(request: Request, exc: ModelLoadError):
    return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": str(exc)})


async def cache_error_handler(request: Request, exc: CacheError):
    # کش نباید کل سیستم را متوقف کند
    return JSONResponse(status_code=200, content={"warning": "cache_bypassed", "detail": str(exc)})
