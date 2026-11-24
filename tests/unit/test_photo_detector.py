import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.ports.text_detector import TextRegion
from app.infrastructure.detection.chakkhu_photo.geometry import group
from app.infrastructure.detection.photo_detector import PhotoTextDetector
from app.infrastructure.ocr.providers.line_provider import LineOCRProvider


def quad(x, y, width, height):
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def detector_with(quads):
    detector = PhotoTextDetector()

    def engine(image, **kwargs):
        assert kwargs == {"use_det": True, "use_cls": False, "use_rec": False}
        return quads, None

    detector._engine = engine
    return detector


def test_detects_moving_text_without_card_or_template():
    image = Image.new("RGB", (800, 600), "white")
    # Two pieces on one line and a separate second line.
    detector = detector_with([quad(90, 70, 80, 25), quad(190, 70, 90, 25), quad(110, 350, 230, 30)])
    regions = detector.detect(image)
    assert len(regions) == 2
    assert all(len(region.alternatives) == 5 for region in regions)
    assert all(region.field_name is None for region in regions)
    assert regions[0].bbox[0] <= 90 and regions[0].bbox[2] >= 280
    assert regions[1].bbox[1] <= 350
    assert not detector.needs_card


def test_coordinates_scale_back_and_remain_inside_input():
    image = Image.new("RGB", (2400, 1801), "white")
    detector = detector_with(np.array([quad(0, 0, 1599, 30)], dtype=float))
    region = detector.detect(image)[0]
    assert region.bbox[2] == image.width
    for crop in [region, *region.alternatives]:
        x1, y1, x2, y2 = crop.bbox
        assert 0 <= x1 < x2 <= image.width
        assert 0 <= y1 < y2 <= image.height


@pytest.mark.parametrize("quads", [None, [], np.empty((0, 4, 2))])
def test_no_text_does_not_invent_full_image_crop(quads):
    assert detector_with(quads).detect(Image.new("RGB", (100, 100))) == []


def test_default_photo_config_and_rejection_of_destructive_card_crop():
    settings = Settings(_env_file=None, DOCUMENT_TEMPLATE="")
    assert settings.TEXT_DETECTOR == "photo"
    assert settings.DOCUMENT_TEMPLATE is None
    with pytest.raises(ValidationError, match="preserve the full image"):
        Settings(_env_file=None, TEXT_DETECTOR="photo", CARD_DETECTOR="opencv")


def test_vertical_labels_and_large_logos_do_not_swallow_horizontal_lines():
    rows = group(
        [
            quad(0, 0, 100, 120),  # large logo beside two headings
            quad(130, 0, 300, 40),
            quad(150, 60, 200, 30),
            quad(440, 0, 30, 400),  # vertical label beside all rows
            quad(130, 200, 280, 45),
            quad(130, 300, 280, 45),
        ]
    )
    assert len(rows) == 6


def test_vertical_crops_try_both_directions_and_keep_original_boxes():
    image = Image.new("RGB", (200, 400), "white")
    region = detector_with([quad(50, 40, 30, 300)]).detect(image)[0]
    crops = [region, *region.alternatives]
    assert len(crops) == 12
    assert {crop.rotation_degrees_ccw for crop in crops} == {-90, 90}
    for crop in crops:
        assert crop.image.width > crop.image.height
        assert crop.bbox[0] <= 50 and crop.bbox[2] >= 80
        assert crop.bbox[1] <= 40 and crop.bbox[3] >= 340


def test_opposite_directions_are_not_voted_together():
    image = Image.new("RGB", (200, 30))
    region = TextRegion(image, [0, 0, 30, 200], rotation_degrees_ccw=-90)
    region.alternatives = [
        TextRegion(image, region.bbox, rotation_degrees_ccw=-90),
        TextRegion(image, region.bbox, rotation_degrees_ccw=90),
        TextRegion(image, region.bbox, rotation_degrees_ccw=90),
    ]
    provider = Provider(detector_with([]))
    provider._predict = lambda images: [
        ("UPRIGHT", 0.98),
        ("UPRIGHT", 0.99),
        ("garbage", 0.4),
        ("garbage", 0.3),
    ]
    block = provider.recognize_regions([region]).blocks[0]
    assert block.text == "UPRIGHT"
    assert block.crop_rotation_degrees_ccw == -90


class Provider(LineOCRProvider):
    name = "fake"

    def load(self):
        pass

    def _predict(self, images):
        assert len(images) == 4
        return [("TEST 123", 0.90), ("TEST 123", 0.91), ("TEST 128", 0.99), ("OTHER", 0.95)]


def test_crop_vote_is_per_line_and_preserves_reader_outputs():
    image = Image.new("RGB", (200, 30))
    region = TextRegion(image, [0, 0, 100, 20])
    region.alternatives = [
        TextRegion(image, [0, 0, 110, 22]),
        TextRegion(image, [0, 0, 120, 24]),
    ]
    other = TextRegion(image, [0, 40, 120, 60])
    result = Provider(detector_with([])).recognize_regions([region, other])
    first, second = result.blocks
    assert first.text == "TEST 123"
    assert first.raw_text == "TEST 128"
    assert first.candidate_texts == ["TEST 123", "TEST 123", "TEST 128"]
    assert first.recognition_method == "crop_vote"
    assert first.bbox == [0, 0, 120, 24]
    assert first.confidence == 0.91
    assert second.text == second.raw_text == "OTHER"
    assert second.recognition_method == "single_crop"
