import cv2
import numpy as np
from PIL import Image

from app.infrastructure.preprocessing.original_processor import OriginalProcessor


class CLAHEProcessor(OriginalProcessor):
    def process(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(super().process(image).convert("L"))
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return Image.fromarray(enhanced).convert("RGB")
