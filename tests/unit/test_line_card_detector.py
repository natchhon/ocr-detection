import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.infrastructure.detection.line_card_detector import find_line_quadrilateral
from app.infrastructure.detection.opencv_card_detector import OpenCVCardDetector


@pytest.mark.parametrize("scale", [1, 2])
def test_recovers_interrupted_outer_boundary_at_multiple_resolutions(scale):
    image = Image.new("RGB", (600 * scale, 800 * scale), "#333333")
    draw = ImageDraw.Draw(image)
    # Synthetic card with a missing top-edge segment, simulating a clip/occlusion.
    segments = [
        [(100, 100), (240, 90)],
        [(330, 83), (500, 70)],
        [(100, 100), (85, 700)],
        [(500, 70), (520, 730)],
        [(85, 700), (520, 730)],
    ]
    for points in segments:
        draw.line([(x * scale, y * scale) for x, y in points], fill="white", width=3 * scale)
    expected = np.array([[100, 100], [500, 70], [520, 730], [85, 700]])
    corners = find_line_quadrilateral(np.asarray(image.convert("L")))
    assert corners is not None
    assert np.max(np.linalg.norm(corners / scale - expected, axis=1)) < 10
    result = OpenCVCardDetector().detect(image)
    assert result.detected
    assert result.confidence is None


def test_rejects_blank_and_only_two_edges():
    image = Image.new("L", (600, 800), 50)
    assert find_line_quadrilateral(np.asarray(image)) is None
    draw = ImageDraw.Draw(image)
    draw.line([(100, 100), (500, 100)], fill=255, width=3)
    draw.line([(100, 100), (100, 700)], fill=255, width=3)
    assert find_line_quadrilateral(np.asarray(image)) is None
