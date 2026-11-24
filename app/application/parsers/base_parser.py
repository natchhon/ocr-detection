from abc import ABC, abstractmethod

from app.domain.entities.ocr_result import OCRBlock


class FieldParser(ABC):
    @abstractmethod
    def parse(self, blocks: list[OCRBlock], required_fields: tuple[str, ...] = ()) -> dict:
        pass
