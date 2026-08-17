"""
نقطه ورود FastAPI Application.
"""

from fastapi import FastAPI
from api.routers import summarize, health
from api.exception_handlers import (
    validation_error_handler,
    inference_error_handler,
    model_load_error_handler,
    cache_error_handler,
)
from arabic_summarizer.exceptions import (
    ValidationError, InferenceError, ModelLoadError, CacheError
)

app = FastAPI(title="Arabic Summarizer API", version="0.1.0")

# Routers
app.include_router(health.router)
app.include_router(summarize.router)

# Exception Handlers
app.add_exception_handler(ValidationError, validation_error_handler)
app.add_exception_handler(InferenceError, inference_error_handler)
app.add_exception_handler(ModelLoadError, model_load_error_handler)
app.add_exception_handler(CacheError, cache_error_handler)
