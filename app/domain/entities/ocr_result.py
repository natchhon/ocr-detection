from dataclasses import dataclass, field


@dataclass
class OCRBlock:
    text: str
    confidence: float | None = None
    bbox: list[int] = field(default_factory=list)
    raw_text: str | None = None
    field_name: str | None = None
    requires_review: bool = False
    recognition_method: str = "single_crop"
    candidate_texts: list[str] = field(default_factory=list)
    crop_rotation_degrees_ccw: int = 0


@dataclass
class OCRResult:
    provider: str

    blocks: list[OCRBlock] = field(default_factory=list)

    processing_time_ms: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:

        return "\n".join(block.text for block in self.blocks)

    @property
    def confidence(self) -> float | None:

        scores = [block.confidence for block in self.blocks if block.confidence is not None]

        if not scores:
            return None

        return sum(scores) / len(scores)
