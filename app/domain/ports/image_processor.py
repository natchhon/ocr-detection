from abc import ABC, abstractmethod

from PIL import Image


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        pass
