"""
اندپوینت اصلی خلاصه‌سازی.
"""

from fastapi import APIRouter, Depends
from api.schemas.request_models import SummarizeRequest
from api.schemas.response_models import SummarizeResponse
from api.dependencies import get_summarizer
from arabic_summarizer import Summarizer

router = APIRouter(tags=["Summarize"])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(
    request: SummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
):
    summary = summarizer.summarize(text=request.text, ratio=request.ratio)
    return SummarizeResponse(summary=summary)
