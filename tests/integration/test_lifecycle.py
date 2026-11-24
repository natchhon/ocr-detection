from fastapi.testclient import TestClient

from app import bootstrap
from app.core.config import Settings
from app.infrastructure.detection.opencv_detector import OpenCVTextDetector
from app.infrastructure.detection.photo_detector import PhotoTextDetector
from app.infrastructure.ocr.providers.line_provider import LineOCRProvider
from app.main import app


def test_startup_loads_selected_model_once_and_closes(monkeypatch):
    events = []

    class Provider(LineOCRProvider):
        name = "mock"

        def load(self):
            events.append("load")

        def close(self):
            events.append("close")

        def _predict(self, images):
            return [("SYNTHETIC", 0.99) for _ in images]

    provider = Provider(OpenCVTextDetector())
    monkeypatch.setattr(bootstrap, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(bootstrap.OCRProviderFactory, "create", lambda settings: provider)
    monkeypatch.setattr(PhotoTextDetector, "load", lambda self: events.append("detector_load"))
    monkeypatch.setattr(PhotoTextDetector, "close", lambda self: events.append("detector_close"))
    with TestClient(app) as client:
        assert events == ["detector_load", "load"]
        health = client.get("/health").json()
        assert health["provider"] == "mock"
        assert health["text_detector"] == "photo"
        assert health["template"] is None
        assert health["roi_fallback_enabled"] is True
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert events == ["detector_load", "load"]
    assert events == ["detector_load", "load", "close", "detector_close"]
