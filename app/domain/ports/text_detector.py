from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class TextRegion:

    image: Image.Image
    bbox: list[int]


class TextDetector(ABC):

    @abstractmethod
    def detect(
        self,
        image: Image.Image
    ) -> list[TextRegion]:
        pass
