from dataclasses import asdict
from time import perf_counter

from PIL import Image

from app.application.errors import TemplateMatchError
from app.application.parsers.base_parser import FieldParser
from app.domain.ports.card_detector import CardDetection, CardDetector
from app.domain.ports.image_processor import ImageProcessor
from app.domain.ports.ocr_provider import OCRProvider
from app.domain.ports.text_detector import TextDetector


class IDCardService:
    def __init__(
        self,
        ocr_provider: OCRProvider,
        image_processor: ImageProcessor | None = None,
        card_detector: CardDetector | None = None,
        text_detector: TextDetector | None = None,
        fallback_detector: TextDetector | None = None,
        parser: FieldParser | None = None,
        allow_roi_fallback: bool = True,
    ):
        self.ocr_provider = ocr_provider
        self.image_processor = image_processor
        self.card_detector = card_detector
        self.text_detector = text_detector
        self.fallback_detector = fallback_detector
        self.parser = parser
        self.allow_roi_fallback = allow_roi_fallback

    def detect(self, image: Image.Image) -> dict:
        start = perf_counter()
        timings = {
            "card_detection": 0.0,
            "perspective_correction": 0.0,
            "roi_extraction": 0.0,
            "text_detection": 0.0,
        }
        prepared = self.image_processor.process(image) if self.image_processor else image
        timings["preprocessing"] = (perf_counter() - start) * 1000
        card = (
            self.card_detector.detect(prepared) if self.card_detector else CardDetection(prepared)
        )
        timings.update(card.timings_ms)
        working = card.image
        warnings = list(card.warnings)
        detector = self.text_detector
        template_id = None
        if detector is not None and detector.needs_card:
            oriented = detector.orient(working)
            if card.detected and detector.matches_geometry(oriented):
                working = oriented
                template_id = detector.template_id
                warnings.append("template_assumed_document_type_unverified")
            else:
                if not self.allow_roi_fallback:
                    raise TemplateMatchError(
                        "card_not_found" if not card.detected else "template_geometry_mismatch",
                        detector.template_id,
                        card.detected,
                    )
                warnings.append("roi_fallback_card_or_geometry_unverified")
                detector = self.fallback_detector
        if detector is None:
            result = self.ocr_provider.recognize(working)
            actual_detector = getattr(
                getattr(self.ocr_provider, "detector", None), "name", "provider_default"
            )
        else:
            stage = perf_counter()
            regions = detector.detect(working)
            timings["roi_extraction" if detector.needs_card else "text_detection"] = (
                perf_counter() - stage
            ) * 1000
            result = self.ocr_provider.recognize_regions(regions)
            actual_detector = detector.name
        timings.update(result.timings_ms)
        stage = perf_counter()
        required = detector.required_fields if template_id else ()
        fields = self.parser.parse(result.blocks, required) if self.parser else {}
        timings["parsing"] = (perf_counter() - stage) * 1000
        total = (perf_counter() - start) * 1000
        timings["total"] = total
        # Geometry and template selection cannot establish document identity.
        return {
            "provider": result.provider,
            "confidence": result.confidence,
            "processing_time_ms": round(total, 2),
            "text": result.text,
            "total_blocks": len(result.blocks),
            "blocks": [asdict(block) for block in result.blocks],
            "fields": fields,
            "document_type": "unknown",
            "candidate_document_type": "student_id_card" if template_id else None,
            "template": template_id,
            "requires_review": True,
            "warnings": warnings or ["document_type_unverified"],
            "timings_ms": {key: round(value, 3) for key, value in timings.items()},
            "card_detection": {
                "detected": card.detected,
                "confidence": card.confidence,
                "corners": card.corners,
                "corner_coordinate_space": "preprocessed_image",
            },
            "diagnostics": {
                "provider": result.provider,
                "text_detector": actual_detector,
                "requested_text_detector": self.text_detector.name
                if self.text_detector
                else actual_detector,
                "preprocessed_size": list(prepared.size),
                "processed_size": list(working.size),
                "bbox_coordinate_space": "rectified_card"
                if card.detected
                else "preprocessed_image",
                "bbox_format": "xyxy_exclusive",
                "card_rotation_degrees_ccw": detector.card_rotation if template_id else 0,
            },
        }
