# معماری پروژه Arabic Summarizer

## لایه‌ها

1. **UI Layer** (Gradio) - رابط گرافیکی دمو
2. **API Layer** (FastAPI) - سرویس‌دهی محلی REST
3. **Core Library** (arabic_summarizer) - منطق اصلی خلاصه‌سازی
4. **Inference Engine** (ONNX Runtime) - اجرای مدل کوانتیزه‌شده

## نکات کلیدی طراحی

- Cache key از متن نرمال‌شده ساخته می‌شود، نه متن خام.
- Tokenization داخل ONNXEngine انجام می‌شود، نه در سطح Pipeline (جداسازی مسئولیت‌ها).
- تمام خطاها از طریق exceptions.py مدیریت و به پاسخ HTTP مناسب تبدیل می‌شوند.

## نقاط توسعه آینده (Future Extensions)

به فایل `future_extensions.md` مراجعه کنید.
