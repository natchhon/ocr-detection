from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.application.parsers.student_id_parser import StudentIDParser
from app.application.services.id_card_service import IDCardService
from app.core.config import Settings, get_settings
from app.domain.ports.ocr_provider import OCRProvider
from app.infrastructure.detection.factory import create_card_detector, create_text_detector
from app.infrastructure.detection.opencv_detector import OpenCVTextDetector
from app.infrastructure.ocr.factory import OCRProviderFactory
from app.infrastructure.preprocessing.factory import create_processor


def create_service(settings: Settings, provider: OCRProvider) -> IDCardService:
    return IDCardService(
        ocr_provider=provider,
        image_processor=create_processor(settings.IMAGE_PROCESSOR, settings.MAX_IMAGE_DIMENSION),
        card_detector=create_card_detector(settings),
        text_detector=create_text_detector(settings),
        fallback_detector=OpenCVTextDetector(),
        allow_roi_fallback=settings.OCR_ROI_FALLBACK,
        parser=StudentIDParser(
            settings.OCR_MIN_CONFIDENCE,
            settings.FIELD_PATTERNS,
            settings.SCHOOL_NAMES,
            settings.CAMPUS_NAMES,
            school_names_en=settings.SCHOOL_NAMES_EN,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    provider = OCRProviderFactory.create(settings)
    # Validate templates before allocating model memory.
    service = create_service(settings, provider)
    try:
        await run_in_threadpool(service.text_detector.load)
        await run_in_threadpool(provider.load)
        app.state.ocr_provider = provider
        app.state.id_card_service = service
        yield
    finally:
        await run_in_threadpool(provider.close)
        await run_in_threadpool(service.text_detector.close)
