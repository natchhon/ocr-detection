"""Repeatable, local-only benchmark. Reports contain metrics, never OCR text."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont

from app.bootstrap import create_service
from app.core.config import Settings
from app.infrastructure.ocr.factory import OCRProviderFactory
from app.infrastructure.preprocessing.factory import create_processor
from app.infrastructure.preprocessing.image_loader import load_image
from scripts.evaluate_accuracy import accuracy_metrics


def synthetic_sample() -> tuple[Image.Image, dict]:
    image = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    lines = ["SYNTHETIC SCHOOL", "TEST00001", "2026 - 2027"]
    for y, text in zip([70, 170, 270], lines):
        draw.text((60, y), text, font=font, fill="black")
    return image, {"text": "\n".join(lines), "fields": {}}


def load_dataset(manifest: Path | None, required_split: str | None = None):
    if manifest is None:
        if required_split:
            raise ValueError("A held-out evaluation manifest is required")
        image, reference = synthetic_sample()
        return [(image, reference)], {
            "kind": "synthetic_english_smoke_only",
            "split": "synthetic",
            "image_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }
    content = manifest.read_bytes()
    data = json.loads(content)
    split = data.get("split")
    if split not in {"tuning", "evaluation"} or (required_split and split != required_split):
        raise ValueError("Dataset split must match the requested tuning/evaluation purpose")
    records = data.get("samples", [])
    if not records:
        raise ValueError("Dataset has no samples")
    samples, seen = [], set()
    for record in records:
        path = (manifest.parent / record["image"]).resolve()
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Dataset image exceeds the file-size limit")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("sha256") != digest:
            raise ValueError("Dataset checksum mismatch; verify the intended original image")
        if digest in seen:
            raise ValueError("Duplicate image in dataset")
        if not isinstance(record.get("text"), str) or not isinstance(
            record.get("fields", {}), dict
        ):
            raise ValueError("Expected manually verified text and field mapping")
        seen.add(digest)
        image, _ = load_image(raw, 20_000_000)
        samples.append((image, record))
    return samples, {
        "kind": "authorized_local_manifest",
        "split": split,
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
    }


def latency_metrics(values: list[float]) -> dict:
    if not values:
        raise ValueError("No timing samples")
    ordered = sorted(values)

    def percentile(p):
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": percentile(0.5) if len(values) >= 5 else None,
        "p95_ms": percentile(0.95) if len(values) >= 100 else None,
        "p99_ms": percentile(0.99) if len(values) >= 1000 else None,
        "timing_samples": len(values),
    }


def write_report(path: Path, report: dict):
    # Explicit output only; owner-readable even for future reports with sensitive metadata.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def benchmark(
    manifest=None,
    provider_name="chakkhu",
    variants=None,
    runs=3,
    warmup=1,
    template=None,
    cpu_threads=None,
    beam=4,
    required_split=None,
    detector="photo",
):
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be positive and warmup nonnegative")
    samples, dataset = load_dataset(manifest, required_split)
    settings = Settings(
        _env_file=None,
        OCR_PROVIDER=provider_name,
        OCR_CPU_THREADS=cpu_threads,
        OCR_BEAM=beam,
        DOCUMENT_TEMPLATE=template,
        TEXT_DETECTOR="roi" if template else detector,
        CARD_DETECTOR="opencv" if template else "none",
    )
    provider = OCRProviderFactory.create(settings)
    # Validate the selected template before loading weights.
    service = create_service(settings, provider)
    start = perf_counter()
    reports = []
    try:
        service.text_detector.load()
        provider.load()
        load_ms = (perf_counter() - start) * 1000
        for variant in variants or ["original"]:
            service.image_processor = create_processor(variant, settings.MAX_IMAGE_DIMENSION)
            for _ in range(warmup):
                for image, _ in samples:
                    service.detect(image)
            latencies, stages, predictions = [], {}, []
            for iteration in range(runs):
                for image, _ in samples:
                    result = service.detect(image)
                    latencies.append(result["processing_time_ms"])
                    for name, value in result["timings_ms"].items():
                        stages.setdefault(name, []).append(value)
                    if iteration == 0:
                        predictions.append(result)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            reports.append(
                {
                    "processor": variant,
                    "latency": latency_metrics(latencies),
                    "stage_mean_ms": {
                        name: statistics.mean(values) for name, values in stages.items()
                    },
                    "accuracy": accuracy_metrics(
                        [reference for _, reference in samples], predictions
                    ),
                    "process_peak_rss_mb": peak
                    / (1024 * 1024 if platform.system() == "Darwin" else 1024),
                }
            )
    finally:
        provider.close()
        service.text_detector.close()
    packages = [
        "pillow",
        "numpy",
        "opencv-python-headless",
        "rapidocr-onnxruntime",
        "torch" if provider_name == "chakkhu" else "onnxruntime",
    ]
    folder = settings.MODEL_DIR / ("chakkhu" if provider_name == "chakkhu" else "chakkhu-onnx")
    weights = (
        ["model.safetensors"]
        if provider_name == "chakkhu"
        else ["encoder.int8.onnx", "decoder.int8.onnx"]
    )
    return {
        "dataset": dataset,
        "unique_images": len(samples),
        "runs": runs,
        "warmup_runs": warmup,
        "provider": provider_name,
        "beam": beam,
        "batch_size": settings.OCR_BATCH_SIZE,
        "cpu_threads_requested": cpu_threads,
        "device": settings.OCR_DEVICE,
        "text_detector": settings.TEXT_DETECTOR,
        "model_load_ms": load_ms,
        "model_sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in weights
        },
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "results": reports,
        "limitations": [
            "CER uses NFC Unicode code points, not Khmer grapheme clusters.",
            "WER uses whitespace tokens; Khmer words without spaces are not segmented.",
            "P50 needs 5 timings; P95 needs 100; P99 needs 1000. Repeats are not independent documents.",
            "RSS is cumulative process high-water memory, not isolated per-variant memory.",
            "Inference timings exclude upload/decode and startup; warmup is separate.",
            "Synthetic English results cannot establish Khmer or real-card accuracy.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--provider", default="chakkhu", choices=["chakkhu", "chakkhu_onnx"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original", "grayscale", "clahe", "khmer"],
        choices=["original", "grayscale", "clahe", "khmer"],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--detector", choices=["photo", "opencv"], default="photo")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.manifest,
        args.provider,
        args.variants,
        args.runs,
        args.warmup,
        args.template,
        args.cpu_threads,
        args.beam,
        detector=args.detector,
    )
    write_report(args.output, report)
    print(
        f"Benchmark saved: {len(report['results'])} configurations, {report['unique_images']} unique images; no recognized text in report."
    )


if __name__ == "__main__":
    main()
