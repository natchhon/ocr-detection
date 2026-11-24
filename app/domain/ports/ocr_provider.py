from abc import ABC, abstractmethod

from PIL import Image

from app.domain.entities.ocr_result import OCRResult


class OCRProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def recognize(
        self,
        image: Image.Image
    ) -> OCRResult:
        pass

    def close(self) -> None:
        pass
