from PIL import Image, ImageOps

from app.domain.ports.image_processor import ImageProcessor


class OriginalProcessor(ImageProcessor):
    def __init__(self, max_dimension: int = 2400):
        self.max_dimension = max_dimension

    def process(self, image: Image.Image) -> Image.Image:
        result = ImageOps.exif_transpose(image).convert("RGB")
        result.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
        return result
