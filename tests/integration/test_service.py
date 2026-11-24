from PIL import Image

from app.application.parsers.student_id_parser import StudentIDParser
from app.application.services.id_card_service import IDCardService
from app.domain.ports.card_detector import CardDetection, CardDetector
from app.domain.ports.text_detector import TextDetector, TextRegion
from app.infrastructure.detection.roi_detector import ROIDetector
from app.infrastructure.detection.template_loader import CardTemplate, RegionSpec
from app.infrastructure.ocr.providers.line_provider import LineOCRProvider


class Card(CardDetector):
    def __init__(self, detected):
        self.detected = detected

    def detect(self, image):
        return CardDetection(image, self.detected)


class Fallback(TextDetector):
    def detect(self, image):
        return [TextRegion(image, [0, 0, image.width, image.height])]


class Provider(LineOCRProvider):
    name = "mock"

    def load(self):
        pass

    def _predict(self, images):
        return [("បេប័មសិក្សា", 0.99) for _ in images]


def service(detected=True, rotation=0):
    template = CardTemplate(
        template_id="synthetic",
        calibrated=True,
        orientation="portrait",
        card_rotation=rotation,
        aspect_ratio_range=(0.4, 0.8),
        regions={"education_level": RegionSpec(bbox=(0, 0, 0.2, 1), rotation=90, required=True)},
    )
    return IDCardService(
        Provider(Fallback()),
        card_detector=Card(detected),
        text_detector=ROIDetector(template),
        fallback_detector=Fallback(),
        parser=StudentIDParser(),
    )


def test_template_pipeline_has_named_fields_but_does_not_claim_identity():
    result = service().detect(Image.new("RGB", (100, 200)))
    assert result["template"] == "synthetic"
    assert result["fields"]["education_level"]["suggested_code"] == "primary"
    assert result["fields"]["education_level"]["value"] == "បេប័មសិក្សា"
    assert result["diagnostics"]["bbox_coordinate_space"] == "rectified_card"
    assert result["document_type"] == "unknown" and result["requires_review"]
    assert result["candidate_document_type"] == "student_id_card"
    assert result["blocks"][0]["bbox"] == [0, 0, 20, 200]


def test_card_failure_and_wrong_geometry_use_unnamed_fallback():
    for pipeline, size in [(service(False), (100, 200)), (service(True), (200, 100))]:
        result = pipeline.detect(Image.new("RGB", size))
        assert result["template"] is None and result["fields"] == {}
        assert result["diagnostics"]["text_detector"] == "opencv"
        assert "roi_fallback_card_or_geometry_unverified" in result["warnings"]


def test_template_orientation_and_empty_required_field():
    pipeline = service(rotation=90)
    pipeline.ocr_provider._predict = lambda images: [(" ", 0.99)]
    result = pipeline.detect(Image.new("RGB", (200, 100)))
    assert result["diagnostics"]["processed_size"] == [100, 200]
    assert result["fields"]["education_level"]["source"] == "missing"
    assert result["blocks"] == []


def test_strict_roi_rejects_failed_match_without_running_any_ocr():
    import pytest

    from app.application.errors import TemplateMatchError

    for pipeline, size, reason in [
        (service(False), (100, 200), "card_not_found"),
        (service(True), (200, 100), "template_geometry_mismatch"),
    ]:
        pipeline.allow_roi_fallback = False
        pipeline.ocr_provider.recognize = lambda image: pytest.fail("OCR must not run")
        pipeline.ocr_provider.recognize_regions = lambda regions: pytest.fail("OCR must not run")
        pipeline.fallback_detector.detect = lambda image: pytest.fail("Fallback must not run")
        with pytest.raises(TemplateMatchError) as caught:
            pipeline.detect(Image.new("RGB", size))
        assert caught.value.reason == reason
        assert caught.value.template == "synthetic"


def test_strict_roi_allows_matching_template():
    pipeline = service()
    pipeline.allow_roi_fallback = False
    result = pipeline.detect(Image.new("RGB", (100, 200)))
    assert result["template"] == "synthetic"
    assert result["diagnostics"]["text_detector"] == "roi"


def test_default_photo_pipeline_preserves_full_image_and_never_labels_ais_fields():
    from app.bootstrap import create_service
    from app.core.config import Settings

    pipeline = create_service(Settings(_env_file=None), Provider(Fallback()))
    assert pipeline.card_detector is None
    pipeline.text_detector.detect = lambda image: [
        TextRegion(image.crop((10, 100, 160, 130)), [10, 100, 160, 130])
    ]
    result = pipeline.detect(Image.new("RGB", (200, 300)))
    assert result["diagnostics"]["processed_size"] == [200, 300]
    assert result["diagnostics"]["text_detector"] == "photo"
    assert result["diagnostics"]["bbox_coordinate_space"] == "preprocessed_image"
    assert result["template"] is None
    assert result["fields"] == {}
    assert not result["card_detection"]["detected"]
    assert result["blocks"][0]["bbox"] == [10, 100, 160, 130]
    assert result["blocks"][0]["field_name"] is None
