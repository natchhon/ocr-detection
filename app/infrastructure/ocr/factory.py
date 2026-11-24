from app.core.config import Settings
from app.domain.ports.ocr_provider import OCRProvider
from app.infrastructure.detection.opencv_detector import OpenCVTextDetector


class OCRProviderFactory:
    @staticmethod
    def create(settings: Settings) -> OCRProvider:
        options = dict(
            detector=OpenCVTextDetector(),
            min_confidence=settings.OCR_MIN_CONFIDENCE,
            batch_size=settings.OCR_BATCH_SIZE,
            beam=settings.OCR_BEAM,
            ctc_weight=settings.OCR_CTC_WEIGHT,
            cpu_threads=settings.OCR_CPU_THREADS,
            padding=settings.OCR_TEXT_PADDING,
            min_text_height=settings.OCR_MIN_TEXT_HEIGHT,
        )
        if settings.OCR_PROVIDER == "chakkhu":
            from app.infrastructure.ocr.providers.chakkhu import ChakkhuProvider

            return ChakkhuProvider(settings.MODEL_DIR / "chakkhu", settings.OCR_DEVICE, **options)
        if settings.OCR_PROVIDER == "chakkhu_onnx":
            from app.infrastructure.ocr.providers.chakkhu_onnx import ChakkhuONNXProvider

            return ChakkhuONNXProvider(settings.MODEL_DIR / "chakkhu-onnx", **options)
        raise ValueError(f"Unsupported OCR provider: {settings.OCR_PROVIDER}")
