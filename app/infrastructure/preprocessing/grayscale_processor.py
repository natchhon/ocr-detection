from PIL import Image

from app.infrastructure.preprocessing.original_processor import OriginalProcessor


class GrayscaleProcessor(OriginalProcessor):
    def process(self, image: Image.Image) -> Image.Image:
        return super().process(image).convert("L").convert("RGB")
