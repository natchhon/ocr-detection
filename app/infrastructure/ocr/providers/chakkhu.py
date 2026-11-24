import sys
import tempfile

from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image

from app.domain.ports.ocr_provider import OCRProvider
from app.domain.ports.text_detector import TextDetector

from app.domain.entities.ocr_result import (
    OCRBlock,
    OCRResult
)


class ChakkhuProvider(OCRProvider):

    def __init__(
        self,
        model_path: Path,
        device: str,
        detector: TextDetector,
        min_confidence: float = 0.80
    ):

        self.model_path = model_path.resolve()
        self.device = device
        self.detector = detector
        self.min_confidence = min_confidence

        self.model = None
        self.tokenizer = None
        self.read = None

        self.lock = Lock()

    @property
    def name(self) -> str:
        return "chakkhu"

    def load(self) -> None:

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}"
            )

        if str(self.model_path) not in sys.path:
            sys.path.insert(0, str(self.model_path))

        from chakkhu import load, read

        self.read = read

        self.model, self.tokenizer = load(
            str(self.model_path),
            device=self.device
        )

    def recognize(
        self,
        image: Image.Image
    ) -> OCRResult:

        if self.model is None or self.read is None:
            raise RuntimeError("Chakkhu model is not loaded")

        start = perf_counter()

        regions = self.detector.detect(image)

        blocks = []

        if regions:

            with tempfile.TemporaryDirectory() as tmp:

                paths = []

                for index, region in enumerate(regions):

                    path = Path(tmp) / f"{index}.png"

                    region.image.save(path)

                    paths.append(str(path))

                with self.lock:

                    predictions = self.read(
                        self.model,
                        self.tokenizer,
                        paths,
                        confidence=True
                    )

                if len(predictions) != len(regions):
                    raise RuntimeError(
                        "OCR result count mismatch"
                    )

                for region, prediction in zip(
                    regions, predictions
                ):

                    text, confidence = prediction

                    text = text.strip()
                    confidence = float(confidence)

                    # Ignore empty text
                    if not text:
                        continue

                    # Ignore low-confidence results
                    if confidence < self.min_confidence:
                        continue

                    blocks.append(
                        OCRBlock(
                            text=text,
                            confidence=confidence,
                            bbox=region.bbox
                        )
                    )

        return OCRResult(
            provider=self.name,
            blocks=blocks,
            processing_time_ms=(
                perf_counter() - start
            ) * 1000
        )

    def close(self) -> None:

        with self.lock:

            self.model = None
            self.tokenizer = None
            self.read = None
