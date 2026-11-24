"""Explicit local debug export of rectified image, boxes, and rotated ROI crops."""

import argparse
import io
import os
from pathlib import Path

from PIL import ImageDraw

from app.infrastructure.detection.opencv_card_detector import OpenCVCardDetector
from app.infrastructure.detection.photo_detector import PhotoTextDetector
from app.infrastructure.detection.roi_detector import ROIDetector
from app.infrastructure.detection.template_loader import load_template
from app.infrastructure.preprocessing.factory import create_processor
from app.infrastructure.preprocessing.image_loader import load_image


def save_private(image, path):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New private directory; exports may contain student information",
    )
    args = parser.parse_args()
    if args.image.stat().st_size > 10 * 1024 * 1024:
        parser.error("Image exceeds file-size limit")
    image, _ = load_image(args.image.read_bytes(), 20_000_000)
    image = create_processor("original").process(image)
    working = image
    detector = PhotoTextDetector()
    if args.template:
        card = OpenCVCardDetector().detect(image)
        if not card.detected:
            parser.error("No card found; refusing template ROI export")
        detector = ROIDetector(load_template(args.template))
        working = detector.orient(card.image)
    try:
        regions = detector.detect(working)
    finally:
        detector.close()
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_private(working, args.output_dir / "processed.png")
    overlay = working.copy()
    draw = ImageDraw.Draw(overlay)
    for index, region in enumerate(regions):
        draw.rectangle(region.bbox, outline="red", width=2)
        draw.text((region.bbox[0], region.bbox[1]), str(index), fill="red")
        save_private(region.image, args.output_dir / f"roi-{index:02d}.png")
    save_private(overlay, args.output_dir / "boxes.png")
    print(f"Exported {len(regions)} regions locally. No text recognition was run.")


if __name__ == "__main__":
    main()
