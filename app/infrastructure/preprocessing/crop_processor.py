from PIL import Image, ImageOps

from app.domain.ports.image_processor import ImageProcessor


class CropProcessor(ImageProcessor):
    def __init__(self, padding: int = 0, min_height: int = 0):
        self.padding = padding
        self.min_height = min_height

    def process(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.min_height and image.height < self.min_height:
            scale = min(4.0, self.min_height / image.height)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return ImageOps.expand(image, border=self.padding, fill="white")
