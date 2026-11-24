import cv2
import numpy as np

from PIL import Image

from app.domain.ports.text_detector import (
    TextDetector,
    TextRegion
)


class OpenCVTextDetector(TextDetector):

    def detect(
        self,
        image: Image.Image
    ) -> list[TextRegion]:

        image_width, image_height = image.size

        gray = cv2.cvtColor(
            np.array(image.convert("RGB")),
            cv2.COLOR_RGB2GRAY
        )

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, image_width // 70), 3)
        )

        merged = cv2.dilate(
            binary,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            # Ignore small regions
            if w < 30 or h < 8:
                continue

            # Ignore large non-text regions
            if h > image_height * 0.20:
                continue

            # Ignore extremely large regions
            if (w * h) > (image_width * image_height * 0.15):
                continue

            # Ignore non-horizontal regions
            if w / h < 1.5:
                continue

            bbox = [
                max(0, x - 5),
                max(0, y - 5),
                min(image_width, x + w + 5),
                min(image_height, y + h + 5)
            ]

            regions.append(
                TextRegion(
                    image=image.crop(tuple(bbox)),
                    bbox=bbox
                )
            )

        return sorted(
            regions,
            key=lambda region: (
                region.bbox[1],
                region.bbox[0]
            )
        )
