from dataclasses import asdict

from PIL import Image

from app.domain.ports.ocr_provider import OCRProvider


class IDCardService:

    def __init__(
        self,
        ocr_provider: OCRProvider
    ):

        self.ocr_provider = ocr_provider

    def detect(self, image: Image.Image) -> dict:

        result = self.ocr_provider.recognize(image)

        return {
            "provider": result.provider,
            "confidence": result.confidence,
            "processing_time_ms": round(
                result.processing_time_ms, 2
            ),
            "text": result.text,
            "total_blocks": len(result.blocks),
            "blocks": [
                asdict(block)
                for block in result.blocks
            ]
        }
