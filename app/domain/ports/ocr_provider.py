from abc import ABC, abstractmethod

from PIL import Image

from app.domain.entities.ocr_result import OCRResult
from app.domain.ports.text_detector import TextRegion


class OCRProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def recognize(self, image: Image.Image) -> OCRResult:
        pass

    def close(self) -> None:
        pass

    def recognize_regions(self, regions: list[TextRegion]) -> OCRResult:
        """Optional line-recognition capability for application-selected regions."""
        raise NotImplementedError("This provider does not support supplied regions")
