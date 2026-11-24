from app.core.config import Settings
from app.domain.ports.ocr_provider import OCRProvider


class OCRProviderFactory:

    @staticmethod
    def create(settings: Settings) -> OCRProvider:

        provider = settings.OCR_PROVIDER.lower()

        match provider:

            case "chakkhu":

                from app.infrastructure.ocr.providers.chakkhu import (
                    ChakkhuProvider
                )

                from app.infrastructure.detection.opencv_detector import (
                    OpenCVTextDetector
                )

                return ChakkhuProvider(
                    model_path=settings.MODEL_DIR / "chakkhu",
                    device=settings.OCR_DEVICE,
                    detector=OpenCVTextDetector(),
                    min_confidence=settings.OCR_MIN_CONFIDENCE
                )

            case _:
                raise ValueError(
                    f"Unsupported OCR provider: {provider}"
                )
