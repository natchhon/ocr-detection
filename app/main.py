from fastapi import FastAPI, Request

from app.api.v1.endpoints.id_card import router
from app.bootstrap import lifespan
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health(request: Request):
    service = request.app.state.id_card_service
    detector = service.text_detector
    return {
        "status": "UP",
        "provider": request.app.state.ocr_provider.name,
        "device": settings.OCR_DEVICE,
        "text_detector": detector.name if detector else "provider_default",
        "template": detector.template_id if detector else None,
        "roi_fallback_enabled": service.allow_roi_fallback,
    }
