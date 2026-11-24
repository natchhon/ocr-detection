"""NFC code-point CER, whitespace-token WER, and exact field metrics."""

import argparse
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from app.application.parsers.text_normalizer import normalize_text


def accuracy_metrics(references: list[dict], predictions: list[dict]) -> dict:
    if len(references) != len(predictions) or not references:
        raise ValueError("Expected equal, non-empty reference and prediction sets")
    chars = words = char_errors = word_errors = 0
    fields = {}
    for reference, prediction in zip(references, predictions):
        expected = normalize_text(reference["text"])
        actual = normalize_text(prediction["text"])
        chars += len(expected)
        char_errors += Levenshtein.distance(expected, actual)
        expected_words, actual_words = expected.split(), actual.split()
        words += len(expected_words)
        word_errors += Levenshtein.distance(expected_words, actual_words)
        for name, value in reference.get("fields", {}).items():
            hits, total = fields.get(name, (0, 0))
            predicted = prediction.get("fields", {}).get(name, {}).get("value")
            hits += predicted is not None and normalize_text(str(predicted)) == normalize_text(
                str(value)
            )
            fields[name] = (hits, total + 1)
    total_fields = sum(total for _, total in fields.values())
    return {
        "cer": char_errors / chars if chars else None,
        "wer_whitespace": word_errors / words if words else None,
        "character_errors": char_errors,
        "reference_characters": chars,
        "word_errors": word_errors,
        "reference_whitespace_tokens": words,
        "field_exact_match": sum(hits for hits, _ in fields.values()) / total_fields
        if total_fields
        else None,
        "fields": {
            name: {"exact_match": hits / total, "samples": total}
            for name, (hits, total) in fields.items()
        },
        "samples": len(references),
    }


def main():
    from scripts.benchmark import benchmark, write_report

    parser = argparse.ArgumentParser(
        description="Evaluate a fixed, held-out local manifest; never uploads images."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="chakkhu", choices=["chakkhu", "chakkhu_onnx"])
    parser.add_argument("--template", type=Path)
    parser.add_argument("--detector", choices=["photo", "opencv"], default="photo")
    args = parser.parse_args()
    report = benchmark(
        args.manifest,
        args.provider,
        ["original", "grayscale", "clahe", "khmer"],
        runs=1,
        template=args.template,
        required_split="evaluation",
        detector=args.detector,
    )
    write_report(args.output, report)


if __name__ == "__main__":
    main()
