import inspect
from pathlib import Path

from PIL import Image

from app.domain.ports.text_detector import TextDetector
from app.infrastructure.ocr.providers.line_provider import LineOCRProvider
from app.infrastructure.ocr.providers.local_module import load_local_module


class ChakkhuProvider(LineOCRProvider):
    def __init__(
        self,
        model_path: Path,
        device: str,
        detector: TextDetector,
        min_confidence: float = 0.8,
        **options,
    ):
        super().__init__(detector, min_confidence, **options)
        self.model_path = model_path.resolve()
        self.device = device
        self.model = self.tokenizer = self.read = None

    @property
    def name(self) -> str:
        return "chakkhu"

    def load(self) -> None:
        with self.lock:
            if self.model is not None:
                return
            for name in (
                "chakkhu/__init__.py",
                "model.safetensors",
                "config.json",
                "tokenizer.json",
            ):
                if not (self.model_path / name).is_file():
                    raise FileNotFoundError(f"Missing local Chakkhu file: {name}")
            import torch

            if self.cpu_threads is not None:
                torch.set_num_threads(self.cpu_threads)
            module = load_local_module(
                "_id_card_chakkhu", self.model_path / "chakkhu/__init__.py", package=True
            )
            required = {"beam", "lam", "batch", "confidence"}
            if not required.issubset(inspect.signature(module.read).parameters):
                raise RuntimeError(
                    "Unsupported Chakkhu read API; use the documented pinned revision"
                )
            model, tokenizer = module.load(str(self.model_path), device=self.device)
            self.model, self.tokenizer, self.read = model, tokenizer, module.read

    def _predict(self, images: list[Image.Image]) -> list:
        if self.model is None or self.read is None:
            raise RuntimeError("Chakkhu model is not loaded")
        import torch

        with torch.inference_mode():
            return self.read(
                self.model,
                self.tokenizer,
                images,
                confidence=True,
                batch=self.batch_size,
                beam=self.beam,
                lam=self.ctc_weight,
            )

    def close(self) -> None:
        with self.lock:
            self.model = self.tokenizer = self.read = None
