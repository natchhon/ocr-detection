from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep

import pytest
from PIL import Image

from app.core.config import Settings
from app.domain.ports.text_detector import TextDetector, TextRegion
from app.infrastructure.ocr.factory import OCRProviderFactory
from app.infrastructure.ocr.providers.chakkhu import ChakkhuProvider
from app.infrastructure.ocr.providers.chakkhu_onnx import ChakkhuONNXProvider
from app.infrastructure.ocr.providers.line_provider import LineOCRProvider
from app.infrastructure.ocr.providers.netra_onnx import NetraONNXProvider


class Detector(TextDetector):
    def detect(self, image):
        return [TextRegion(image, [0, 0, image.width, image.height], "name_kh")]


class FakeProvider(LineOCRProvider):
    name = "fake"

    def load(self):
        pass

    def _predict(self, images):
        return [("  e\u0301  ", 0.4) for _ in images]


def test_retains_uncertain_raw_and_normalized_text():
    result = FakeProvider(Detector()).recognize(Image.new("RGB", (30, 20)))
    block = result.blocks[0]
    assert block.text == "é" and block.raw_text == "  e\u0301  "
    assert block.requires_review and block.field_name == "name_kh"
    assert block.bbox == [0, 0, 30, 20]
    assert "recognition" in result.timings_ms


@pytest.mark.parametrize("prediction", [[("text", float("nan"))], [("text", 2)], [("text",)], []])
def test_rejects_invalid_predictions(prediction):
    provider = FakeProvider(Detector())
    provider._predict = lambda images: prediction
    with pytest.raises(RuntimeError):
        provider.recognize(Image.new("RGB", (30, 20)))


def test_empty_predictions_and_no_regions():
    provider = FakeProvider(Detector())
    provider._predict = lambda images: [(" ", 0.9)]
    assert provider.recognize(Image.new("RGB", (30, 20))).blocks == []
    provider._predict = lambda images: pytest.fail("Must not call inference for no regions")
    assert provider.recognize_regions([]).blocks == []


def test_provider_serializes_inference():
    provider = FakeProvider(Detector())
    guard = Lock()

    def predict(images):
        assert guard.acquire(blocking=False)
        try:
            sleep(0.01)
            return [("test", 0.9)]
        finally:
            guard.release()

    provider._predict = predict
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(provider.recognize, [Image.new("RGB", (30, 20))] * 3))
    assert len(results) == 3


@pytest.mark.parametrize(
    "name,expected", [("chakkhu", ChakkhuProvider), ("chakkhu_onnx", ChakkhuONNXProvider), ("netra_onnx", NetraONNXProvider)]
)
def test_factory_and_missing_models_fail_locally(name, expected, tmp_path):
    provider = OCRProviderFactory.create(
        Settings(_env_file=None, OCR_PROVIDER=name, MODEL_DIR=tmp_path)
    )
    assert isinstance(provider, expected)
    with pytest.raises(FileNotFoundError):
        provider.load()


def test_pytorch_forwards_supported_parameters_and_images():
    provider = ChakkhuProvider(
        Path("models/chakkhu"), "cpu", Detector(), batch_size=3, beam=0, ctc_weight=0.3
    )
    provider.model = object()

    def read(model, tokenizer, images, **kwargs):
        assert all(isinstance(image, Image.Image) for image in images)
        assert kwargs == {"confidence": True, "batch": 3, "beam": 0, "lam": 0.3}
        return [("TEST", 0.9)]

    provider.read = read
    assert provider.recognize(Image.new("RGB", (30, 20))).text == "TEST"


def test_onnx_forwards_supported_parameters_without_torch(tmp_path):
    provider = ChakkhuONNXProvider(tmp_path, Detector(), batch_size=2, beam=0, cpu_threads=2)

    class Model:
        def read(self, images, **kwargs):
            assert isinstance(images[0], Image.Image)
            assert kwargs == {"confidence": True, "batch": 2, "beam": 0, "lam": 0.5}
            return [("TEST", 0.9)]

    provider.model = Model()
    assert provider.recognize(Image.new("RGB", (30, 20))).text == "TEST"
    provider.close()
    assert provider.model is None
