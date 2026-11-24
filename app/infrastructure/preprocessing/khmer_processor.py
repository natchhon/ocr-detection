import cv2
import numpy as np
from PIL import Image

from app.infrastructure.preprocessing.clahe_processor import CLAHEProcessor


class KhmerProcessor(CLAHEProcessor):
    """Experimental mild denoising; no thresholding or morphological erosion."""

    def process(self, image: Image.Image) -> Image.Image:
        enhanced = np.asarray(super().process(image))
        return Image.fromarray(cv2.bilateralFilter(enhanced, 3, 15, 15))
