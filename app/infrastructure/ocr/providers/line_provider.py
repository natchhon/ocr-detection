import math
from abc import abstractmethod
from threading import Lock
from time import perf_counter

from PIL import Image

from app.application.parsers.text_normalizer import normalize_text
from app.domain.entities.ocr_result import OCRBlock, OCRResult
from app.domain.ports.ocr_provider import OCRProvider
from app.domain.ports.text_detector import TextDetector, TextRegion
from app.infrastructure.detection.chakkhu_photo.geometry import vote
from app.infrastructure.preprocessing.crop_processor import CropProcessor


class LineOCRProvider(OCRProvider):
    def __init__(
        self,
        detector: TextDetector,
        min_confidence: float = 0.8,
        batch_size: int = 16,
        beam: int = 4,
        ctc_weight: float = 0.5,
        cpu_threads: int | None = None,
        padding: int = 0,
        min_text_height: int = 0,
    ):
        self.detector = detector
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.beam = beam
        self.ctc_weight = ctc_weight
        self.cpu_threads = cpu_threads
        self.crop_processor = CropProcessor(padding, min_text_height)
        self.lock = Lock()

    def recognize(self, image: Image.Image) -> OCRResult:
        start = perf_counter()
        regions = self.detector.detect(image)
        detection_ms = (perf_counter() - start) * 1000
        result = self.recognize_regions(regions)
        result.timings_ms["text_detection"] = detection_ms
        result.processing_time_ms = (perf_counter() - start) * 1000
        return result

    def recognize_regions(self, regions: list[TextRegion]) -> OCRResult:
        start = perf_counter()
        groups = [[region, *region.alternatives] for region in regions]
        crops = [self.crop_processor.process(item.image) for group in groups for item in group]
        crop_ms = (perf_counter() - start) * 1000
        waiting = perf_counter()
        with self.lock:
            queue_ms = (perf_counter() - waiting) * 1000
            inference_start = perf_counter()
            predictions = self._predict(crops) if crops else []
            inference_ms = (perf_counter() - inference_start) * 1000
        if len(predictions) != len(crops):
            raise RuntimeError("OCR result count mismatch")
        validated = []
        for prediction in predictions:
            if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
                raise RuntimeError("Invalid OCR prediction format")
            raw, score = prediction
            if not isinstance(raw, str):
                raise RuntimeError("Invalid OCR prediction text")
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("Invalid OCR confidence")
            validated.append((raw, score))
        blocks = []
        offset = 0
        for region, group in zip(regions, groups):
            readings = validated[offset : offset + len(group)]
            offset += len(group)
            voted = len(group) > 1
            orientations = {}
            for index, crop in enumerate(group):
                orientations.setdefault(crop.rotation_degrees_ccw, []).append(index)
            choices = []
            for indices in orientations.values():
                current = [readings[index] for index in indices]
                text, score = vote(current) if voted else current[0]
                best = max(indices, key=lambda index: readings[index][1])
                choices.append((text, score, best))
            text, score, best = max(choices, key=lambda choice: choice[1])
            raw = readings[best][0]
            text = normalize_text(text)
            # Match the demo's Photo-mode suppression of tiny, uncertain fragments.
            if voted and len(text) <= 2 and score < 0.6:
                continue
            if text:
                blocks.append(
                    OCRBlock(
                        text,
                        score,
                        group[best].bbox,
                        raw,
                        region.field_name,
                        score < self.min_confidence,
                        "crop_vote" if voted else "single_crop",
                        [reading[0] for reading in readings] if voted else [],
                        group[best].rotation_degrees_ccw,
                    )
                )
        return OCRResult(
            self.name,
            blocks,
            (perf_counter() - start) * 1000,
            {
                "crop_preprocessing": crop_ms,
                "recognition_queue": queue_ms,
                "recognition": inference_ms,
            },
        )

    @abstractmethod
    def _predict(self, images: list[Image.Image]) -> list:
        pass
