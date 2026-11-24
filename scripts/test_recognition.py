"""Inspect one local image. Full OCR output is saved only when explicitly requested."""

import argparse
import json
from pathlib import Path

from app.bootstrap import create_service
from app.core.config import Settings
from app.infrastructure.ocr.factory import OCRProviderFactory
from app.infrastructure.preprocessing.image_loader import load_image
from scripts.benchmark import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--provider", choices=["chakkhu", "chakkhu_onnx"], default="chakkhu")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--detector", choices=["photo", "opencv"], default="photo")
    parser.add_argument(
        "--processor", choices=["original", "grayscale", "clahe", "khmer"], default="original"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit opt-in to saving personal OCR text; use a protected local path",
    )
    args = parser.parse_args()
    settings = Settings(
        _env_file=None,
        OCR_PROVIDER=args.provider,
        IMAGE_PROCESSOR=args.processor,
        DOCUMENT_TEMPLATE=args.template,
        TEXT_DETECTOR="roi" if args.template else args.detector,
        CARD_DETECTOR="opencv" if args.template else "none",
    )
    if args.image.stat().st_size > settings.MAX_FILE_SIZE:
        parser.error("Image exceeds file-size limit")
    image, diagnostics = load_image(args.image.read_bytes(), settings.MAX_IMAGE_PIXELS)
    provider = OCRProviderFactory.create(settings)
    service = create_service(settings, provider)
    try:
        provider.load()
        result = service.detect(image)
    finally:
        provider.close()
        service.text_detector.close()
    result["diagnostics"].update(diagnostics)
    if args.output:
        write_report(args.output, result)
    print(
        json.dumps(
            {
                "diagnostics": result["diagnostics"],
                "timings_ms": result["timings_ms"],
                "total_blocks": result["total_blocks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
