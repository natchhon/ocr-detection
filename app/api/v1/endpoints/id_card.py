import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_id_card_service
from app.application.errors import TemplateMatchError
from app.application.services.id_card_service import IDCardService
from app.core.config import get_settings
from app.infrastructure.preprocessing.image_loader import ImageValidationError, load_image

router = APIRouter(prefix="/id-card", tags=["ID Card"])
logger = logging.getLogger(__name__)


@router.post("/detect")
async def detect(
    response: Response,
    file: UploadFile = File(...),
    service: IDCardService = Depends(get_id_card_service),
):
    start = perf_counter()
    settings = get_settings()
    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    content = await file.read(settings.MAX_FILE_SIZE + 1)
    await file.close()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    try:
        image, diagnostics = await run_in_threadpool(load_image, content, settings.MAX_IMAGE_PIXELS)
    except ImageValidationError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    validation_ms = (perf_counter() - start) * 1000
    try:
        result = await run_in_threadpool(service.detect, image)
    except TemplateMatchError as exc:
        logger.warning("ocr_template_not_matched request_id=%s reason=%s", request_id, exc.reason)
        raise HTTPException(
            422,
            detail={
                "code": "template_not_matched",
                "reason": exc.reason,
                "message": str(exc),
                "template": exc.template,
                "card_detected": exc.card_detected,
                "inference_performed": False,
                "diagnostics": {
                    **diagnostics,
                    "request_id": request_id,
                    "provider": service.ocr_provider.name,
                    "requested_text_detector": "roi",
                    "text_detector": "not_run",
                },
            },
            headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
        ) from None
    except Exception as exc:
        # Model exceptions can contain OCR text or paths. Do not log their messages.
        logger.error("ocr_failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        raise HTTPException(
            503, "OCR processing failed", headers={"X-Request-ID": request_id}
        ) from None
    total_ms = (perf_counter() - start) * 1000
    result["diagnostics"].update(diagnostics, request_id=request_id)
    result["timings_ms"].update(image_validation=round(validation_ms, 3), total=round(total_ms, 3))
    result["processing_time_ms"] = round(total_ms, 2)
    logger.info(
        "ocr_completed request_id=%s width=%s height=%s format=%s bytes=%s sha256=%s provider=%s detector=%s duration_ms=%.2f",
        request_id,
        diagnostics["width"],
        diagnostics["height"],
        diagnostics["format"],
        diagnostics["file_size"],
        diagnostics["sha256"],
        result["provider"],
        result["diagnostics"]["text_detector"],
        total_ms,
    )
    return {"success": True, "data": result}
