from dataclasses import dataclass, field


@dataclass
class OCRBlock:

    text: str
    confidence: float | None = None
    bbox: list[int] = field(default_factory=list)


@dataclass
class OCRResult:

    provider: str

    blocks: list[OCRBlock] = field(
        default_factory=list
    )

    processing_time_ms: float = 0.0

    @property
    def text(self) -> str:

        return "\n".join(
            block.text for block in self.blocks
        )

    @property
    def confidence(self) -> float | None:

        scores = [
            block.confidence
            for block in self.blocks
            if block.confidence is not None
        ]

        if not scores:
            return None

        return sum(scores) / len(scores)
