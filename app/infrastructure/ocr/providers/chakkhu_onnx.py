import inspect
from pathlib import Path

from PIL import Image

from app.domain.ports.text_detector import TextDetector
from app.infrastructure.ocr.providers.line_provider import LineOCRProvider
from app.infrastructure.ocr.providers.local_module import load_local_module


class ChakkhuONNXProvider(LineOCRProvider):
    """Opt-in CPU INT8 adapter for upstream revision 8d7373487dfb56e4f13ba26cea31e1f858bd6ddd."""

    def __init__(
        self, model_path: Path, detector: TextDetector, min_confidence: float = 0.8, **options
    ):
        super().__init__(detector, min_confidence, **options)
        self.model_path = model_path.resolve()
        self.model = None

    @property
    def name(self) -> str:
        return "chakkhu_onnx"

    def load(self) -> None:
        with self.lock:
            if self.model is not None:
                return
            for name in (
                "chakkhu_onnx.py",
                "khnormal.py",
                "encoder.int8.onnx",
                "decoder.int8.onnx",
                "tokenizer.json",
            ):
                if not (self.model_path / name).is_file():
                    raise FileNotFoundError(
                        f"Missing local ONNX file: {name}; run the explicit model download command first"
                    )
            # Upstream uses an absolute khnormal import; reject collisions with another source.
            load_local_module("khnormal", self.model_path / "khnormal.py")
            module = load_local_module("_id_card_chakkhu_onnx", self.model_path / "chakkhu_onnx.py")
            if not {"beam", "lam", "batch", "confidence"}.issubset(
                inspect.signature(module.Chakkhu.read).parameters
            ):
                raise RuntimeError("Unsupported Chakkhu ONNX read API")
            self.model = module.Chakkhu(
                repo=str(self.model_path), int8=True, threads=self.cpu_threads or 0
            )

    def _predict(self, images: list[Image.Image]) -> list:
        if self.model is None:
            raise RuntimeError("Chakkhu ONNX model is not loaded")
        return self.model.read(
            images, confidence=True, batch=self.batch_size, beam=self.beam, lam=self.ctc_weight
        )

    def close(self) -> None:
        with self.lock:
            self.model = None
