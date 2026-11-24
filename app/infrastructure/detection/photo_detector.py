"""Local Chakkhu demo Photo-mode detection; no card template or network calls."""

import math
from threading import Lock

import numpy as np
from PIL import Image

from app.domain.ports.text_detector import TextDetector, TextRegion
from app.infrastructure.detection.chakkhu_photo.geometry import (
    PADS,
    cut_straight,
    cut_tilted,
    group,
)


class PhotoTextDetector(TextDetector):
    name = "photo"

    def __init__(self):
        self._engine = None
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self._engine is None:
                from rapidocr_onnxruntime import RapidOCR

                # The pinned wheel contains the detector weights. No runtime download.
                self._engine = RapidOCR(
                    det_box_thresh=0.3,
                    det_unclip_ratio=1.6,
                    intra_op_num_threads=2,
                    inter_op_num_threads=1,
                )

    def close(self) -> None:
        with self._lock:
            self._engine = None

    def detect(self, image: Image.Image) -> list[TextRegion]:
        self.load()
        small = image.convert("RGB")
        small.thumbnail((1600, 1600))
        with self._lock:
            quads, _ = self._engine(np.asarray(small), use_det=True, use_cls=False, use_rec=False)
        # Use separate axis scales: thumbnail rounding need not preserve an exact ratio.
        scale = np.array([image.width / small.width, image.height / small.height])
        quads = [] if quads is None else [np.asarray(q) * scale for q in quads]
        pixels = np.asarray(image.convert("RGB"))
        regions = []
        for row in group(quads):
            x1, y1, x2, y2 = row["box"]
            vertical = y2 - y1 > 1.5 * (x2 - x1)
            working = pixels
            if vertical:
                # Read both directions, without a school-specific rotation rule.
                working = np.rot90(pixels, k=-1)
                rotated = [np.column_stack([image.height - q[:, 1], q[:, 0]]) for q in row["quads"]]
                points = np.concatenate(rotated)
                row = {
                    "quads": rotated,
                    "box": [*points.min(axis=0), *points.max(axis=0)],
                }
            candidates = []
            for padding in PADS:
                for cut in (cut_straight, cut_tilted):
                    crop, polygon = cut(working, row, padding, 1.0)
                    if crop.shape[0] < 8 or crop.shape[1] < 8:
                        continue
                    if vertical:
                        polygon = [(y, image.height - x) for x, y in polygon]
                    xs, ys = zip(*polygon)
                    box = [
                        max(0, math.floor(min(xs))),
                        max(0, math.floor(min(ys))),
                        min(image.width, math.ceil(max(xs))),
                        min(image.height, math.ceil(max(ys))),
                    ]
                    if box[2] > box[0] and box[3] > box[1]:
                        candidate = Image.fromarray(crop)
                        candidates.append(
                            TextRegion(candidate, box, rotation_degrees_ccw=-90 if vertical else 0)
                        )
                        if vertical:
                            candidates.append(
                                TextRegion(candidate.rotate(180), box, rotation_degrees_ccw=90)
                            )
            if candidates:
                candidates[0].alternatives = candidates[1:]
                regions.append(candidates[0])
        return regions
