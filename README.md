# ID Card OCR

Local FastAPI service with interchangeable Chakkhu PyTorch, Chakkhu ONNX INT8, and optional Netra ONNX providers. `/health`, `/docs`, and `POST /api/v1/id-card/detect` remain available.
The default pipeline preserves original RGB input, detects text with local PP-OCRv4,
and reads multiple crop variants with the existing Chakkhu PyTorch model. All model inference runs locally.

## Run and verify

Use the existing `.venv` (currently Python 3.10.12). No model is downloaded at startup.

```bash
make install
make install-dev
make check
make test
make run                       # localhost:8000, one model worker
make dev                       # development only
curl -F 'file=@/private/path/card.png' http://localhost:8000/api/v1/id-card/detect
```

`make install PROVIDER=chakkhu_onnx` installs the optional runtime without requiring
PyTorch. `make download-model PROVIDER=chakkhu_onnx` explicitly downloads the pinned
INT8 model into `models/chakkhu-onnx`. The download command refuses to overwrite a
nonempty model directory. For an isolated download use
`python -m scripts.download_model --provider chakkhu --model-dir /private/new-models`.
Set `OCR_PROVIDER=chakkhu_onnx` to select it. Only the selected provider is loaded.
For Netra, install the compatible CPU runtime with `make install PROVIDER=netra_onnx`,
then run `make download-model PROVIDER=netra_onnx`. This downloads the pinned 41 MB
Netra recognition bundle into `models/netra-onnx/recognition`. Your `.env` already
selects Netra with `OCR_PROVIDER=netra_onnx`. Restart the service after installing
the runtime and model. Netra supplies line recognition while this service retains
its configured text detector. The recognition model is MIT licensed, though its
training data includes a research-only dataset, so check usage terms before commercial deployment.
Surya and PaddleOCR recognition providers are not implemented; PP-OCRv4 is used
only for text detection, while Chakkhu reads the crops. Unknown providers fail validation.

Copy `.env.example` to a local `.env` only after reviewing it. The example includes
all supported pipeline options. The local `TEXT_DETECTOR` setting was migrated
from `opencv` to `photo`; infrastructure files were not changed. CPU thread count is unset by default; beam 4 and batch 16 preserve the
installed reader's defaults. `OCR_BEAM=0` selects the supported CTC-only decoder.
No general Khmer spelling checker is enabled for model inference.

## Deploy on Ubuntu with systemd

These instructions assume Ubuntu 22.04 or newer, an x86-64 CPU server, and a
non-root Linux account that owns the cloned repository. Replace `/opt/id-card-ocr`
and `deploy` with your actual clone path and username. The default PyTorch model
is already tracked in the repository.

```bash
cd /opt/id-card-ocr
sudo apt update
sudo apt install -y python3-venv curl libgl1 libglib2.0-0 libgomp1 fonts-dejavu-core
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements/chakkhu.txt
.venv/bin/python -m pip check
test -e .env || install -m 600 .env.example .env
```

Review `.env` before starting. Create `/etc/systemd/system/id-card-ocr.service`
using `sudoedit`. `User=` must be the existing Linux account that owns the clone.
Check it with `id deploy` after replacing `deploy` with your username. `217/USER`
in the systemd journal means that systemd could not resolve the configured user.

```ini
[Unit]
Description=ID Card OCR API

[Service]
Type=exec
User=deploy
WorkingDirectory=/opt/id-card-ocr
Environment=APP_ENV=production
ExecStart=/opt/id-card-ocr/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Validate the paths and account, then start the service. If it is already restart-looping
with `217/USER`, fix `User=` first and restart the service to clear the loop.

```bash
id deploy
sudo systemd-analyze verify /etc/systemd/system/id-card-ocr.service
sudo systemctl daemon-reload
sudo systemctl reset-failed id-card-ocr
sudo systemctl restart id-card-ocr
sudo systemctl enable id-card-ocr
curl -fsS --retry 10 --retry-connrefused http://127.0.0.1:8000/health
sudo journalctl -u id-card-ocr -n 80 --no-pager
```

For a fictitious upload test, run this in the repository directory.

```bash
smoke_dir="$(mktemp -d)"
.venv/bin/python -m scripts.create_synthetic_dataset "$smoke_dir/fixture"
curl -fsS -F "file=@$smoke_dir/fixture/synthetic.png" \
  http://127.0.0.1:8000/api/v1/id-card/detect
```

The API has no authentication. Keep Uvicorn bound to `127.0.0.1`, and connect
temporarily from your workstation with an SSH tunnel. Keep the tunnel terminal open,
then visit `http://127.0.0.1:8000/docs` locally.

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8000:127.0.0.1:8000 deploy@YOUR_SERVER_IP
```

Stop and disable the service with `sudo systemctl disable --now id-card-ocr`.
For application access over a network, put it behind an authenticated HTTPS reverse
proxy with upload and concurrency limits, while keeping Uvicorn on loopback.

## Architecture and pipeline

`app/domain/ports` defines image preprocessing, card detection, text detection, and
OCR interfaces. `app/application/services` orchestrates the pipeline and
`app/application/parsers` extracts fields without importing OpenCV, Torch, or ONNX.
`app/bootstrap.py` composes adapters through factories. Both providers share line
validation and batching behavior, load once at startup, and serialize inference
with a lock. Chakkhu receives PIL crops directly; no crop files are persisted.

1. Check upload byte limit, decoded JPEG/PNG type, single frame, and pixel limit.
2. Apply EXIF orientation, RGB conversion, and configured maximum dimension.
3. Optionally apply grayscale, CLAHE, or mild denoising (`khmer`). These are
   experiments, not accuracy guarantees; no destructive thresholding is applied
   to recognition crops. Chakkhu itself converts lines to grayscale internally.
4. Detect text anywhere in the original image with PP-OCRv4 through pinned
   `rapidocr-onnxruntime==1.4.4`. Its wheel bundles the weights; no network access
   is needed at startup or for recognition. Card boundary detection is disabled.
5. Group compatible horizontal text pieces conservatively. Keep large logos,
   neighboring baselines and vertical labels separate. Produce straight and
   deskewed crops at three padding levels, as in the Chakkhu demo's Photo mode.
   Try both 90° directions for vertical labels and vote within each direction.
6. Read crop variants with Chakkhu, then combine readings with the demo's
   confidence-weighted word vote. No word checker is used in Photo mode, matching
   the demo. Detection, recognition and voting all run locally.
7. Return text and bounding boxes without assuming a school layout. `fields` is
   empty in Photo mode; personal names and IDs are not assigned from fixed positions.

## Recognition across card layouts

```bash
make run-photo    # recommended: explicitly selects Photo mode, no card crop/template
make run-ais      # compatibility alias for run-photo; no longer enables fixed AIS ROIs
make recognize-ais  # Photo-mode recognition of the original sample
curl -F 'file=@samples/ais_card_v2.jpg' http://localhost:8000/api/v1/id-card/detect
```

Restart an older server before testing. `/health` must show `text_detector: "photo"`
and `template: null`. A response should also show `diagnostics.text_detector: "photo"`,
`template: null`, `card_detection.detected: false`, and bounding boxes in the full
preprocessed image. `make run` respects `.env`; migrate an old `TEXT_DETECTOR=opencv`
or `roi` to `photo`, set `CARD_DETECTOR=none`, and clear `DOCUMENT_TEMPLATE`, or use
`make run-photo` to override those settings explicitly.

This integrates the demo's **Photo path**, with additional grouping safeguards and
vertical crop trials. It does not implement the demo's Auto page/line fallback,
Document export, or automatically guarantee the hosted demo's exact output.
The local recognizer weights are unchanged. See [attribution](THIRD_PARTY_NOTICES.md).

Photo mode handles varying text positions without relying on one sample's measured
coordinates. Blur, glare, stylized logos, short lines, and character confusions can
still produce errors; confidence is not measured accuracy. The old single-sample
6/7 field result in [the calibration report](reports/ais-calibration.md) is historical
and does not describe this pipeline. See [the correction report](reports/photo-pipeline.md).

### Optional legacy modes

`TEXT_DETECTOR=opencv` retains the old contour baseline for comparisons.
Fixed ROI extraction remains an explicit experiment: select `TEXT_DETECTOR=roi`,
`CARD_DETECTOR=opencv`, and a reviewed `DOCUMENT_TEMPLATE` together. The supplied
`configs/templates/ais_student_card.json` is calibrated to one image only. Its
aspect-ratio check does not verify layout or identity and must not be used as a
universal AIS/Aii detector. `OCR_ROI_FALLBACK=false` rejects missing cards or geometry
mismatches before inference; it cannot reject every wrong layout with similar geometry.

## Response compatibility and interpretation

The `success/data` envelope and `provider`, `confidence`, `processing_time_ms`,
`text`, `total_blocks`, and `blocks` keys are retained. Additions include `fields`,
`template`, `warnings`, `requires_review`, `card_detection`, `timings_ms`, and
`diagnostics`. Blocks include `raw_text`, `field_name`, `requires_review`, `recognition_method`,
`candidate_texts`, and `crop_rotation_degrees_ccw`. For a voted line, `text` is the
combined reading, `raw_text` is the highest-confidence crop reading in the selected
orientation, and `candidate_texts` preserves every model-returned crop reading. The
bounding box encloses that selected crop; it does not certify every voted character.

Two intentional behavioral changes:

- Nonempty low-confidence predictions are retained and flagged instead of dropped.
  `OCR_MIN_CONFIDENCE` now determines review status. Empty predictions are omitted.
  Overall confidence is the mean of all returned block scores, not measured accuracy.
- `processing_time_ms` now covers upload validation and the pipeline; the
  `recognition` stage isolates inference. `recognition_queue` reports lock wait.
  Standalone service/benchmark totals exclude HTTP upload/decode. Startup and
  benchmark warmup are measured separately.

Bounding boxes are `[left, top, right, bottom]` with exclusive right/bottom edges.
`diagnostics.bbox_coordinate_space` is `preprocessed_image` or `rectified_card`.
`processed_size` describes that coordinate frame; `preprocessed_size` describes the
EXIF-oriented/resized input. Card corners reference `preprocessed_image` before
rectification. `card_rotation_degrees_ccw` describes subsequent whole-card rotation.
Text crops may rotate, upscale, or gain padding, but their boxes stay in the stated
image frame. Boxes are **not mapped back to original upload pixels**.

`raw_text` preserves one upstream reader output, which already applies its
own Khmer normalization. It is not untouched decoder token output or a guaranteed
transcription of printed text. Application normalization performs NFC and edge
trimming only. General dictionaries never rewrite personal names or ID numbers.
Known education variants, including `បេប័មសិក្សា` and `បឋមសិក្សារ`, preserve `value`
and return `suggestion: "បឋមសិក្សា"` plus `suggested_code: "primary"` for review.
Exact education category matches return `code`. Similarity `suggestion_score` is separate
from OCR confidence. Missing required fields and duplicate alternatives are flagged.

`FIELD_PATTERNS` accepts JSON full-match regexes for student ID, card number, and
academic year only. Patterns must be validated against real business formats; no
format is guessed. Unconfigured identifier formats report `format_valid: null`
and require review. School and campus dictionaries are configurable category lists. `SCHOOL_NAMES_EN`
controls English school-name suggestions. Personal names always require review
(`personal_name_unverified`), even at high model confidence; their values are never
automatically corrected.

## Local inspection, benchmarking, and accuracy

```bash
# Diagnostics only on stdout; no recognized text or images saved by default.
.venv/bin/python -m scripts.test_recognition /private/path/card.png
# Explicit opt-in to private OCR JSON output (contains personal information).
.venv/bin/python -m scripts.test_recognition /private/path/card.png --output private-data/result.json
# Explicit opt-in to image exports; runs detection only, no text recognition.
# Output directory must be new.
.venv/bin/python -m scripts.visualize_detections /private/path/card.png --output-dir private-data/preview
# Add --template /private/path/calibrated-template.json for rotated ROI previews.

make benchmark                         # synthetic English smoke comparison
make benchmark ARGS='--cpu-threads 4 --beam 0'
make benchmark ARGS='--manifest /private/tuning/manifest.json'
make evaluate ARGS='--manifest /private/evaluation/manifest.json'
```

A manifest declares `split: "tuning"` or `"evaluation"` and a nonempty `samples`
list. Each sample contains a relative image path, SHA-256 of its original bytes,
manually verified `text`, and optional `fields` mapping names to exact values.
Checksum mismatches and duplicate images are rejected. `make evaluate` requires
an evaluation split. Keep that fixed held-out set separate from tuning images;
labels alone cannot prove disjointness, so compare hashes when preparing datasets.

```json
{
  "split": "evaluation",
  "samples": [{
    "image": "original.png",
    "sha256": "replace-with-actual-sha256",
    "text": "manually verified text in reading order",
    "fields": {"student_id": "TEST00001"}
  }]
}
```

Use identical manifests for RGB, grayscale, CLAHE, Khmer preprocessing, ROI, decoder,
CPU-thread, and provider comparisons. Add `--template` for ROI benchmarks. Reports
contain aggregate metrics and hashes, not recognized text or image paths. CER uses
NFC code points. WER uses whitespace-delimited tokens, **not Khmer word segmentation**.
Field exact matches use final `value`, never suggested corrections. Student-ID and
Khmer-name scores are reported separately when those labels exist. Missing labels
produce `null` metrics rather than invented accuracy. P50 needs 5 timings, P95 100,
and P99 1000; repeats do not create independent documents. Peak RSS is the cumulative
process high-water mark; use separate processes for provider comparisons.

Generate a fully fictitious card to exercise perspective and ROI mechanics:

```bash
.venv/bin/python -m scripts.create_synthetic_dataset /tmp/ocr-synthetic
make benchmark ARGS='--manifest /tmp/ocr-synthetic/manifest.json --template /tmp/ocr-synthetic/template.json --variants original'
```

This uses the local DejaVu Sans font at `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`.
It is neither an AIS calibration nor a representative Khmer evaluation dataset.
See [the measured implementation report](reports/accuracy-performance.md).

## Models, privacy, and deployment limits

Pinned downloads:

- [PyTorch source](https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer/tree/e3c8ff8ead6a3fa28d39a4f4fa779e92bbc4ad7d)
  at `e3c8ff8ead6a3fa28d39a4f4fa779e92bbc4ad7d` (the installed snapshot).
- [ONNX source](https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer-ONNX/tree/8d7373487dfb56e4f13ba26cea31e1f858bd6ddd)
  at `8d7373487dfb56e4f13ba26cea31e1f858bd6ddd`.

Their published code/model licenses state Apache-2.0; retain attribution to
**Ing Muyleang**, creator of the Chakkhu model and demo, as “Chakkhu by Ing
Muyleang.” If you redistribute the model, code, or adapted Photo-mode components,
retain applicable Apache-2.0 notices and include [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The upstream model card also requests citation of the model. Local model
Python is executable trusted code: deploy only reviewed, immutable directories.
Startup validates required files and supported API parameters, but does not
cryptographically attest arbitrary replacement files. Benchmark reports record
weight hashes. Source/API compatibility does not prove identical training weights
between the two snapshots. PyTorch's line-width cap is 2400; this ONNX reader's is
4096, so long-line comparisons also differ in preprocessing.

No real card was sent to the public demo or any external OCR API. Images and OCR
text are not logged or persisted by the service. Diagnostics include dimensions,
format, bytes, SHA-256, generated request ID, provider, detector, and duration.
Protect those logs and their retention as operational metadata. Debug exports and
benchmark datasets are local opt-ins; `samples/`, `private-data/`, and `artifacts/` are ignored
by Git, but do not place real cards elsewhere in the repository.

The service has **no authentication** and is intended for local development.
Before external deployment, add authenticated access, TLS, request/concurrency
limits, private log/data retention, and review exposure of `/docs` and `/health`.
No public debug web UI was added. Production deployment, model training, learned
card classification, and production accuracy certification are outside this change.

Rollback: restore the previous application revision while preserving your existing
`.env` and PyTorch model. Configurable experiments can be disabled by selecting
`OCR_PROVIDER=chakkhu`, `IMAGE_PROCESSOR=original`, `CARD_DETECTOR=none`, and
`TEXT_DETECTOR=opencv`, with `DOCUMENT_TEMPLATE` unset. This resets pipeline choices;
low-confidence retention remains the new documented API behavior.
