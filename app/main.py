from fastapi import FastAPI, Request

from app.bootstrap import lifespan
from app.core.config import get_settings
from app.api.v1.endpoints.id_card import router


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.include_router(
    router,
    prefix="/api/v1"
)


@app.get("/health")
def health(request: Request):

    return {
        "status": "UP",
        "provider": request.app.state.ocr_provider.name,
        "device": settings.OCR_DEVICE
    }
