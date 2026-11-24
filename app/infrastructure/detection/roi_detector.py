import math

from PIL import Image

from app.domain.ports.text_detector import TextDetector, TextRegion
from app.infrastructure.detection.template_loader import CardTemplate


class ROIDetector(TextDetector):
    name = "roi"
    needs_card = True

    def __init__(self, template: CardTemplate):
        if not template.calibrated:
            raise ValueError("Cannot use an uncalibrated template")
        self.template = template
        self.template_id = template.template_id
        self.card_rotation = template.card_rotation
        self.required_fields = tuple(
            name for name, spec in template.regions.items() if spec.required
        )

    def orient(self, image: Image.Image) -> Image.Image:
        return image.rotate(self.template.card_rotation, expand=True)

    def matches_geometry(self, image: Image.Image) -> bool:
        low, high = self.template.aspect_ratio_range
        portrait = image.height > image.width
        return low <= image.width / image.height <= high and portrait == (
            self.template.orientation == "portrait"
        )

    def detect(self, image: Image.Image) -> list[TextRegion]:
        if not self.matches_geometry(image):
            raise ValueError("Rectified image does not match template geometry")
        regions = []
        for name, spec in self.template.regions.items():
            x1, y1, x2, y2 = spec.bbox
            box = [
                math.floor(x1 * image.width),
                math.floor(y1 * image.height),
                math.ceil(x2 * image.width),
                math.ceil(y2 * image.height),
            ]
            crop = image.crop(tuple(box)).rotate(spec.rotation, expand=True)
            regions.append(TextRegion(crop, box, name))
        return regions
