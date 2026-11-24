"""Create a reproducible fictitious card fixture and its known ROI template, never an AIS template."""

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.benchmark import write_report
from scripts.visualize_detections import save_private


def create_dataset(output: Path):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    card = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(card)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    lines = [
        ("school_name_en", "SYNTHETIC SCHOOL", 70),
        ("student_id", "TEST00001", 170),
        ("academic_year", "2026 - 2027", 270),
    ]
    regions = {}
    for name, text, y in lines:
        draw.text((60, y), text, font=font, fill="black")
        regions[name] = {
            "bbox": [50 / 800, (y - 5) / 500, 500 / 800, (y + 45) / 500],
            "required": True,
            "language": "english",
        }
    # Photograph placeholder deliberately outside all OCR regions.
    draw.rectangle((580, 90, 740, 340), fill="#777777")
    photo = Image.new("RGB", (1000, 700), "#333333")
    photo.paste(card, (100, 100))
    path = output / "synthetic.png"
    save_private(photo, path)
    write_report(
        output / "template.json",
        {
            "template_id": "synthetic_landscape_v1",
            "document_type": "student_id_card",
            "calibrated": True,
            "orientation": "landscape",
            "aspect_ratio_range": [1.4, 1.8],
            "regions": regions,
        },
    )
    write_report(
        output / "manifest.json",
        {
            "split": "tuning",
            "samples": [
                {
                    "image": "synthetic.png",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "text": "\n".join(text for _, text, _ in lines),
                    "fields": {name: text for name, text, _ in lines},
                }
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for private local fixtures")
    create_dataset(parser.parse_args().output)


if __name__ == "__main__":
    main()
