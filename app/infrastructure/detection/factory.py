from app.core.config import Settings
from app.infrastructure.detection.opencv_detector import OpenCVTextDetector
from app.infrastructure.detection.roi_detector import ROIDetector
from app.infrastructure.detection.template_loader import load_template


def create_text_detector(settings: Settings):
    if settings.TEXT_DETECTOR == "photo":
        from app.infrastructure.detection.photo_detector import PhotoTextDetector

        return PhotoTextDetector()
    if settings.TEXT_DETECTOR == "opencv":
        return OpenCVTextDetector()
    return ROIDetector(load_template(settings.DOCUMENT_TEMPLATE))


def create_card_detector(settings: Settings):
    if settings.CARD_DETECTOR == "none":
        return None
    from app.infrastructure.detection.opencv_card_detector import OpenCVCardDetector

    return OpenCVCardDetector()
