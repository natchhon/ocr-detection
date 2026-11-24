from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from app.domain.ports.card_detector import CardDetection, CardDetector
from app.infrastructure.detection.line_card_detector import find_line_quadrilateral
from app.infrastructure.preprocessing.perspective_processor import ordered_corners, rectify


class OpenCVCardDetector(CardDetector):
    """Geometric candidate detector, never a document-type classifier."""

    def detect(self, image: Image.Image) -> CardDetection:
        start = perf_counter()
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        warnings = []
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
            warnings.append("low_detail_or_blur")
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            area = cv2.contourArea(quad)
            if not 0.20 <= area / (image.width * image.height) <= 0.98:
                continue
            corners = ordered_corners(quad)
            lengths = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
            ratio = max(lengths) / max(1, min(lengths))
            if min(lengths) < 40 or ratio > 2.3:
                continue
            candidates.append((area, corners))
        if not candidates:
            quad = find_line_quadrilateral(gray)
            if quad is not None:
                candidates.append((cv2.contourArea(quad), quad))
                warnings.append("line_based_card_candidate")
        detection_ms = (perf_counter() - start) * 1000
        if not candidates:
            return CardDetection(
                image,
                warnings=warnings + ["card_not_found"],
                timings_ms={"card_detection": detection_ms, "perspective_correction": 0.0},
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        # Multiple comparable rectangles are ambiguous: do not choose an arbitrary card.
        if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * 0.8:
            return CardDetection(
                image,
                warnings=warnings + ["ambiguous_card_candidates"],
                timings_ms={"card_detection": detection_ms, "perspective_correction": 0.0},
            )
        corners = candidates[0][1]
        start = perf_counter()
        result = rectify(image, corners)
        return CardDetection(
            result,
            detected=True,
            corners=corners.tolist(),
            warnings=warnings + ["geometry_only_document_type_unverified"],
            timings_ms={
                "card_detection": detection_ms,
                "perspective_correction": (perf_counter() - start) * 1000,
            },
        )
