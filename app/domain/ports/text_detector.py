from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class TextRegion:
    image: Image.Image
    bbox: list[int]
    field_name: str | None = None
    alternatives: list["TextRegion"] = field(default_factory=list)
    rotation_degrees_ccw: int = 0


class TextDetector(ABC):
    name = "opencv"
    needs_card = False
    template_id: str | None = None
    card_rotation: int = 0
    required_fields: tuple[str, ...] = ()

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def orient(self, image: Image.Image) -> Image.Image:
        return image

    def matches_geometry(self, image: Image.Image) -> bool:
        return True

    @abstractmethod
    def detect(self, image: Image.Image) -> list[TextRegion]:
        pass
