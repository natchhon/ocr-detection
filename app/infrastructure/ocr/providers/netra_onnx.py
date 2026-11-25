from pathlib import Path
from app.domain.ports.text_detector import TextDetector

from PIL import Image

from app.infrastructure.ocr.providers.line_provider import LineOCRProvider


class NetraONNXProvider(LineOCRProvider):
    name = "netra_onnx"

    def __init__(self, model_dir: Path, detector: TextDetector, **options):
        super().__init__(detector, **options)
        self.model_dir = model_dir
        self.model = None

    def load(self):
        from netra_ocr.inference.recognizer import Recognizer

        bundle_dir = self.model_dir / "recognition"
        if not (bundle_dir / "netra-model.json").is_file():
            bundle_dir = bundle_dir / "recognition"
        if not (bundle_dir / "netra-model.json").is_file():
            raise FileNotFoundError(f"Netra model bundle not found at {self.model_dir}")
        self.model = Recognizer(bundle_dir, device="cpu")

    def _predict(self, images: list[Image.Image]) -> list:
        if self.model is None:
            raise RuntimeError("Netra recognizer is not loaded")
        return [(result.text, result.confidence) for result in self.model.batch(images, self.batch_size)]

    def close(self):
        self.model = None
