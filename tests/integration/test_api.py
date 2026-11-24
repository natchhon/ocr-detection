import hashlib
import io
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_id_card_service
from app.api.v1.endpoints import id_card
from app.application.services.id_card_service import IDCardService
from app.core.config import Settings
from app.domain.entities.ocr_result import OCRBlock, OCRResult
from app.domain.ports.ocr_provider import OCRProvider
from app.infrastructure.preprocessing.factory import create_processor


class MockProvider(OCRProvider):
    name = "mock"

    def load(self):
        pass

    def recognize(self, image):
        return OCRResult(
            "mock", [OCRBlock("PRIVATE_SYNTHETIC_NAME", 0.95, [0, 0, image.width, image.height])]
        )


@pytest.fixture
def client(monkeypatch):
    settings = Settings(_env_file=None, MAX_FILE_SIZE=1024, MAX_IMAGE_PIXELS=10000)
    monkeypatch.setattr(id_card, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(id_card.router, prefix="/api/v1")
    service = IDCardService(MockProvider(), image_processor=create_processor("original"))
    app.dependency_overrides[get_id_card_service] = lambda: service
    return TestClient(app), service


def encoded(size=(40, 30), fmt="PNG", exif=False):
    image = Image.new("RGB", size, "white")
    if exif:
        image.getexif()[274] = 6
    output = io.BytesIO()
    image.save(output, format=fmt, exif=image.getexif())
    return output.getvalue()


def test_success_preserves_contract_and_private_logs(client, caplog):
    http, _ = client
    content = encoded()
    with caplog.at_level(logging.INFO, logger=id_card.__name__):
        response = http.post(
            "/api/v1/id-card/detect", files={"file": ("private-name.png", content, "image/png")}
        )
    assert response.status_code == 200
    data = response.json()["data"]
    assert {
        "provider",
        "confidence",
        "processing_time_ms",
        "text",
        "total_blocks",
        "blocks",
    } <= data.keys()
    assert data["diagnostics"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert data["diagnostics"]["width"] == 40 and data["diagnostics"]["height"] == 30
    assert data["diagnostics"]["request_id"] == response.headers["X-Request-ID"]
    assert data["document_type"] == "unknown" and data["requires_review"]
    assert "PRIVATE_SYNTHETIC_NAME" not in caplog.text and "private-name.png" not in caplog.text
    assert response.headers["Cache-Control"] == "no-store"
    assert data["timings_ms"]["total"] >= data["timings_ms"]["image_validation"]


@pytest.mark.parametrize(
    "content,code",
    [
        (b"not image", 400),
        (b"x" * 1025, 413),
        (encoded((120, 120)), 413),
        (encoded(fmt="GIF"), 415),
        (b"", 400),
    ],
)
def test_upload_rejections(client, content, code):
    response = client[0].post(
        "/api/v1/id-card/detect", files={"file": ("test", content, "image/png")}
    )
    assert response.status_code == code


def test_exif_preserves_upload_dimensions_and_processed_coordinates(client):
    content = encoded(fmt="JPEG", exif=True)
    response = client[0].post("/api/v1/id-card/detect", files={"file": ("test.jpg", content)})
    data = response.json()["data"]
    assert data["diagnostics"]["width"] == 40
    assert data["diagnostics"]["processed_size"] == [30, 40]
    assert data["blocks"][0]["bbox"] == [0, 0, 30, 40]


def test_exception_does_not_expose_recognized_text(client, caplog):
    def fail(image):
        raise RuntimeError("PRIVATE_SYNTHETIC_NAME")

    client[1].ocr_provider.recognize = fail
    with caplog.at_level(logging.ERROR):
        response = client[0].post("/api/v1/id-card/detect", files={"file": ("test.png", encoded())})
    assert response.status_code == 503
    assert "PRIVATE_SYNTHETIC_NAME" not in response.text + caplog.text


def test_template_failure_is_422_with_input_diagnostics(client, caplog):
    from app.application.errors import TemplateMatchError

    def fail(image):
        raise TemplateMatchError("card_not_found", "synthetic", False)

    client[1].detect = fail
    content = encoded()
    with caplog.at_level(logging.WARNING):
        response = client[0].post(
            "/api/v1/id-card/detect", files={"file": ("private-name.png", content)}
        )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "template_not_matched"
    assert detail["reason"] == "card_not_found"
    assert not detail["inference_performed"]
    assert detail["diagnostics"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert detail["diagnostics"]["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"
    assert "private-name.png" not in caplog.text + response.text
    assert "PRIVATE_SYNTHETIC_NAME" not in caplog.text + response.text
