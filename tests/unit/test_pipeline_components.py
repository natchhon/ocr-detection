from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from app.application.parsers.student_id_parser import StudentIDParser
from app.application.parsers.text_normalizer import normalize_text
from app.core.config import Settings
from app.domain.entities.ocr_result import OCRBlock
from app.infrastructure.detection.opencv_card_detector import OpenCVCardDetector
from app.infrastructure.detection.roi_detector import ROIDetector
from app.infrastructure.detection.template_loader import CardTemplate, RegionSpec, load_template
from app.infrastructure.preprocessing.crop_processor import CropProcessor
from app.infrastructure.preprocessing.factory import create_processor
from app.infrastructure.preprocessing.perspective_processor import ordered_corners, rectify


@pytest.mark.parametrize(
    "kwargs",
    [
        {"OCR_BATCH_SIZE": 0},
        {"OCR_MIN_CONFIDENCE": 1.1},
        {"OCR_CPU_THREADS": 0},
        {"OCR_BEAM": -1},
        {"TEXT_DETECTOR": "roi"},
        {"OCR_PROVIDER": "unknown"},
        {"OCR_PROVIDER": "chakkhu_onnx", "OCR_DEVICE": "cuda"},
        {"FIELD_PATTERNS": {"student_id": "["}},
        {"FIELD_PATTERNS": {"name_kh": ".*"}},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


@pytest.mark.parametrize("variant", ["original", "grayscale", "clahe", "khmer"])
def test_preprocessing_preserves_dimensions_and_source(variant):
    image = Image.new("RGB", (300, 200), (150, 100, 40))
    result = create_processor(variant, 150).process(image)
    assert result.size == (150, 100)
    assert result.mode == "RGB"
    assert image.size == (300, 200)


def test_exif_orientation_and_crop_padding():
    image = Image.new("RGB", (40, 20), "red")
    image.getexif()[274] = 6
    assert create_processor("original").process(image).size == (20, 40)
    assert CropProcessor(2, 40).process(image).size == (84, 44)


@pytest.mark.parametrize(
    "box", [[0, 0, 0, 0], [-0.1, 0, 1, 1], [0, 0, 1.1, 1], [0, 0, float("nan"), 1]]
)
def test_invalid_roi(box):
    with pytest.raises(ValidationError):
        RegionSpec(bbox=box)


@pytest.mark.parametrize("rotation", [-90, 90])
def test_roi_rotation_direction_and_coordinates(rotation):
    template = CardTemplate(
        template_id="synthetic",
        calibrated=True,
        orientation="portrait",
        aspect_ratio_range=(0.4, 0.8),
        regions={"education_level": RegionSpec(bbox=(0, 0, 0.2, 1), rotation=rotation)},
    )
    image = Image.new("RGB", (100, 200), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 19, 99), fill="red")
    region = ROIDetector(template).detect(image)[0]
    assert region.bbox == [0, 0, 20, 200]
    assert region.image.size == (200, 20)
    expected_left = (255, 0, 0) if rotation == 90 else (255, 255, 255)
    assert region.image.getpixel((10, 10)) == expected_left
    assert region.field_name == "education_level"


def test_uncalibrated_template_rejected(tmp_path):
    import json

    path = tmp_path / "uncalibrated.json"
    path.write_text(
        json.dumps(
            {
                "template_id": "uncalibrated",
                "calibrated": False,
                "orientation": "portrait",
                "aspect_ratio_range": [0.5, 0.9],
                "regions": {},
            }
        )
    )
    with pytest.raises(ValueError, match="uncalibrated"):
        load_template(path)


def test_ais_template_has_all_fields_and_clockwise_vertical_crop():
    template = load_template(Path("configs/templates/ais_student_card.json"))
    assert template.calibrated and len(template.regions) == 9
    assert template.regions["education_level"].rotation == -90
    assert template.regions["student_id"].required


def test_english_school_error_requires_review_even_at_high_confidence():
    parser = StudentIDParser(school_names_en=["American Intercon School"])
    text = "Americon Iatercon School"
    field = parser.parse([OCRBlock(text, 0.99, field_name="school_name_en")])["school_name_en"]
    assert field["value"] == field["raw_text"] == text
    assert field["suggestion"] == "American Intercon School"
    assert field["requires_review"]


def test_perspective_corner_colors():
    image = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 50, 60), fill="red")
    draw.rectangle((150, 230, 180, 260), fill="blue")
    points = np.array([[180, 260], [20, 30], [20, 260], [180, 30]])
    result = rectify(image, points)
    assert result.size == (160, 230)
    assert result.getpixel((5, 5)) == (255, 0, 0)
    assert result.getpixel((150, 220)) == (0, 0, 255)
    with pytest.raises(ValueError):
        ordered_corners(np.zeros((4, 2)))


def test_card_candidate_and_safe_fallback():
    image = Image.new("RGB", (600, 600), "#333333")
    ImageDraw.Draw(image).polygon([(160, 50), (450, 70), (470, 550), (130, 530)], fill="white")
    result = OpenCVCardDetector().detect(image)
    assert result.detected and len(result.corners) == 4
    assert result.confidence is None
    assert "geometry_only_document_type_unverified" in result.warnings
    blank = OpenCVCardDetector().detect(Image.new("RGB", (200, 300), "white"))
    assert not blank.detected and "card_not_found" in blank.warnings


def test_normalization_and_names_unchanged():
    assert normalize_text(" e\u0301 ") == "é"
    value = "បេប័មសិក្សា"
    parsed = StudentIDParser().parse([OCRBlock(value, 0.99, field_name="name_kh")])
    assert parsed["name_kh"]["value"] == value
    assert "suggestion" not in parsed["name_kh"]
    assert parsed["name_kh"]["requires_review"]
    assert parsed["name_kh"]["review_reasons"] == ["personal_name_unverified"]


@pytest.mark.parametrize("value", ["បេប័មសិក្សា", "បឋមសិក្សារ"])
def test_education_suggestion_preserves_original(value):
    field = StudentIDParser().parse(
        [OCRBlock(value, 0.99, raw_text=value, field_name="education_level")]
    )["education_level"]
    assert field["raw_text"] == field["value"] == value
    assert field["suggestion"] == "បឋមសិក្សា"
    assert field["suggested_code"] == "primary"
    assert field["requires_review"]


def test_field_validation_missing_unknown_confidence_and_duplicates():
    parser = StudentIDParser(patterns={"student_id": r"TEST\d{5}"})
    fields = parser.parse(
        [
            OCRBlock("TEST00001", 0.99, field_name="student_id"),
            OCRBlock("សាកល្បង", None, field_name="name_kh"),
        ],
        ("card_number",),
    )
    assert not fields["student_id"]["requires_review"]
    assert fields["name_kh"]["requires_review"]
    assert fields["card_number"]["source"] == "missing"
    bad = parser.parse([OCRBlock("TESTOOO01", 0.99, field_name="student_id")])
    assert bad["student_id"]["value"] == "TESTOOO01"
    assert bad["student_id"]["requires_review"]
    duplicate = parser.parse(
        [OCRBlock("A", 0.99, field_name="name_en"), OCRBlock("B", 0.99, field_name="name_en")]
    )
    assert duplicate["name_en"]["requires_review"] and duplicate["name_en"]["alternatives"]
