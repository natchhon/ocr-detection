from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class CardDetection:
    image: Image.Image
    detected: bool = False
    confidence: float | None = None
    corners: list[list[float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class CardDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> CardDetection:
        pass
