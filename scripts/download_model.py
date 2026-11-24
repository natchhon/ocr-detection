"""Explicit model download at a reviewed revision; never runs during API startup."""

import argparse
from pathlib import Path

MODELS = {
    "chakkhu": {
        "repo": "muyleanging/Chakkhu-OCR-Khmer",
        "revision": "e3c8ff8ead6a3fa28d39a4f4fa779e92bbc4ad7d",
        "patterns": [
            "chakkhu/*.py",
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "README.md",
        ],
        "folder": "chakkhu",
    },
    "chakkhu_onnx": {
        "repo": "muyleanging/Chakkhu-OCR-Khmer-ONNX",
        "revision": "8d7373487dfb56e4f13ba26cea31e1f858bd6ddd",
        "patterns": [
            "chakkhu_onnx.py",
            "khnormal.py",
            "config.json",
            "tokenizer.json",
            "*.int8.onnx",
            "README.md",
        ],
        "folder": "chakkhu-onnx",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="chakkhu", choices=MODELS)
    parser.add_argument("--model-dir", default=Path("models"), type=Path)
    args = parser.parse_args()
    model = MODELS[args.provider]
    target = args.model_dir / model["folder"]
    if target.exists() and any(target.iterdir()):
        parser.error(
            "Destination is not empty. Keep the working model; use a separate --model-dir for a new download."
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        model["repo"],
        revision=model["revision"],
        local_dir=target,
        allow_patterns=model["patterns"],
    )
    (target / ".model_revision").write_text(model["revision"] + "\n", encoding="utf-8")
    print("Pinned model downloaded. Review local model source and license before use.")


if __name__ == "__main__":
    main()
