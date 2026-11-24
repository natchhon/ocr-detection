# Codex Agent Prompt — ID Card OCR Accuracy and Performance Optimization

## 1. Agent role

Act as a senior Python, Computer Vision, Machine Learning, and Backend Software Engineer.

Your task is to review, refactor, optimize, and extend an existing ID card OCR service while maintaining Clean Architecture, modularity, testability, and support for interchangeable OCR models.

Do not rebuild the entire project unnecessarily. Inspect the existing implementation before making changes.

## 2. Project context

Project directory:

`/App2/id-card-ocr`

Current technology stack:

- Python
- FastAPI
- Uvicorn
- PyTorch
- Chakkhu OCR
- OpenCV
- Pillow
- Pydantic Settings
- Hugging Face Hub
- Makefile
- Pytest

Current OCR model:

https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer

Model demonstration:

https://huggingface.co/spaces/muyleanging/Chakkhu-OCR-Demo

Future supported providers:

- Chakkhu PyTorch
- Chakkhu ONNX INT8
- Surya OCR
- PaddleOCR
- Other OCR models

The application is already running successfully and exposes:

- GET `/health`
- POST `/api/v1/id-card/detect`
- GET `/docs`

Do not introduce breaking changes to existing API contracts without documenting them.

## 3. Existing problems

The current OCR implementation has the following issues:

1. Khmer text recognition contains incorrect characters and words.
2. OpenCV contour detection may merge multiple text lines.
3. Some text regions contain photographs, logos, borders, or unrelated objects.
4. Vertical Khmer text requires rotation before recognition.
5. Perspective distortion affects card layout.
6. OCR confidence can be high even when recognized text is incorrect.
7. Processing time is too high.
8. The current implementation returns raw OCR blocks instead of structured student ID card fields.
9. Previous API responses contained browser screenshot text instead of the intended original card image.
10. The project needs reusable preprocessing, detection, and parsing strategies.

Observed performance:

- UAT server: approximately 19,456 ms.
- Another test: approximately 57,027 ms.

These results came from screenshot-like inputs. Do not treat them as representative student ID card benchmarks.

## 4. Khmer recognition requirements

Known OCR correction examples:

| Incorrect OCR                 | Expected text                 |
| ----------------------------- | ----------------------------- |
| សាធារោគ អន្តរទ្រីប អាមេរិកាំង | សាលារៀន អន្តរទ្វីប អាមេរិកាំង |
| បេប័មសិក្សា                   | បឋមសិក្សា                     |

The education level field is printed vertically on the student ID card.

The user also supplied the spelling `បឋមសិក្សារ`. Preserve printed source text separately from the normalized education-level value.

Do not automatically correct personal names using a general Khmer dictionary.

Distinguish between:

- Unicode normalization errors.
- OCR character recognition errors.
- Incorrect text segmentation.
- Incorrect image orientation.
- Language-specific spelling errors.

Use UTF-8 throughout the application and NFC Unicode normalization where appropriate.

## 5. Architectural requirements

Maintain the existing architecture:

```text
app/
├── core/
├── domain/
│   ├── entities/
│   └── ports/
├── application/
│   ├── services/
│   └── parsers/
├── infrastructure/
│   ├── ocr/
│   │   ├── factory.py
│   │   └── providers/
│   ├── detection/
│   └── preprocessing/
└── api/
```

Use:

- Dependency Injection
- Strategy Pattern
- Factory Pattern
- Adapter Pattern
- Single Responsibility Principle
- Dependency Inversion Principle

The application layer must not depend directly on Chakkhu, PyTorch, OpenCV, or ONNX Runtime.

Keep model-specific logic inside infrastructure adapters.

Do not load all OCR models simultaneously.

## 6. Phase 1 — Review and establish baseline

Inspect:

- Existing source code.
- Git status and uncommitted changes.
- Python version.
- Installed dependencies.
- Model directory.
- Chakkhu model revision.
- Chakkhu recognition API signature.
- Model preprocessing implementation.
- Supported inference parameters.
- CPU and GPU capabilities.
- Existing API response contracts.

Verify that the actual uploaded image matches the intended file.

Add privacy-safe input diagnostics:

- Image width and height.
- Image format.
- File size.
- SHA-256 checksum.
- Request ID.
- Selected OCR provider.
- Selected text detector.
- Processing duration.

Do not log personal information or image contents.

Create a baseline benchmark before making performance or accuracy changes.

## 7. Phase 2 — Improve image preprocessing

Implement independently configurable image preprocessing strategies.

Suggested files:

```text
app/domain/ports/image_processor.py

app/infrastructure/preprocessing/
├── factory.py
├── original_processor.py
├── grayscale_processor.py
├── clahe_processor.py
├── khmer_processor.py
└── perspective_processor.py
```

Support:

- EXIF orientation correction.
- RGB image conversion.
- Grayscale conversion.
- CLAHE contrast enhancement.
- Noise reduction.
- Text padding.
- Small-text upscaling.
- Perspective correction.
- Rotation correction.
- Configurable maximum image resolution.

Avoid aggressive thresholding that removes Khmer vowels, subscripts, and diacritics.

Compare preprocessing variants against identical ground-truth text.

Do not assume that enhanced grayscale always performs better than the original RGB image.

## 8. Phase 3 — Implement card detection

Separate physical card detection from text-line detection.

Suggested implementation:

```text
app/domain/ports/card_detector.py

app/infrastructure/detection/
├── card_detector.py
├── opencv_card_detector.py
├── opencv_detector.py
└── factory.py
```

Requirements:

- Identify card boundaries.
- Detect four card corners when possible.
- Apply perspective transformation.
- Normalize card orientation.
- Support portrait and landscape layouts.
- Reject or flag low-quality images.
- Provide a fallback when card detection fails.
- Do not silently classify arbitrary documents as student ID cards.

Return card detection confidence separately from OCR confidence where available.

## 9. Phase 4 — Implement template-based ROI detection

Implement Region of Interest detection for consistent AIS student ID card layouts.

Suggested structure:

```text
configs/
└── templates/
    └── ais_student_card.json

app/infrastructure/detection/
├── roi_detector.py
├── template_loader.py
└── template_registry.py
```

Support the following fields:

- school_name_kh
- school_name_en
- education_level
- name_kh
- name_en
- student_id
- campus
- card_number
- academic_year

Skip OCR processing for:

- Student photograph.
- Logos.
- Decorative graphics.
- Card clips.
- Background objects.
- Card borders.

Store ROI coordinates as normalized values between 0 and 1 relative to the rectified card image.

Example template format:

```json
{
  "template_id": "ais_student_card_v1",
  "document_type": "student_id_card",
  "orientation": "portrait",
  "regions": {
    "education_level": {
      "bbox": [0.0, 0.0, 0.0, 0.0],
      "rotation": -90,
      "language": "khmer",
      "required": false
    }
  }
}
```

The coordinates above are placeholders, not calibrated regions.

Calculate actual coordinates using authorized sample images after perspective correction.

Verify the vertical text orientation visually. Test both clockwise and counterclockwise rotation if necessary.

Do not hardcode pixel coordinates for one particular photograph into production logic.

## 10. Phase 5 — Improve Chakkhu provider

Review and improve:

`app/infrastructure/ocr/providers/chakkhu.py`

Requirements:

- Load the model once during startup.
- Preserve thread safety.
- Batch text-line recognition.
- Avoid redundant model initialization.
- Use in-memory crops if supported by the actual model API.
- Otherwise, use securely managed temporary files.
- Validate prediction counts.
- Preserve field names and bounding boxes.
- Separate raw OCR text from normalized text.
- Preserve uncertain predictions for review.
- Remove empty predictions from final output.
- Support configurable confidence thresholds.
- Support configurable decoder parameters when supported.

Inspect the actual Chakkhu API before passing parameters.

Evaluate:

- Default decoding.
- CTC-only decoding.
- Supported beam-search configurations.
- Supported batch-size configurations.

Do not invent model parameters.

## 11. Phase 6 — Structured field extraction

Implement:

```text
app/application/parsers/
├── base_parser.py
├── student_id_parser.py
├── text_normalizer.py
├── education_level_parser.py
└── field_validator.py
```

Keep field parsing independent of the OCR provider.

Education level mapping:

```python
EDUCATION_LEVELS = {
    "មត្តេយ្យសិក្សា": "kindergarten",
    "បឋមសិក្សា": "primary",
    "អនុវិទ្យាល័យ": "lower_secondary",
    "វិទ្យាល័យ": "upper_secondary",
}
```

Use controlled dictionaries for:

- School names.
- Campus names.
- Education levels.

Use configurable regular expressions for:

- Student ID.
- Card number.
- Academic year.

Validate against actual business formats before enforcing strict regex patterns.

Use RapidFuzz for suggesting corrections to known categorical values.

Never silently modify personal names or identification numbers.

For uncertain results, return the original value, correction suggestion, confidence, and review status.

## 12. Phase 7 — Performance optimization

Measure the processing time of each stage independently:

- Image validation.
- Image preprocessing.
- Card detection.
- Perspective correction.
- ROI extraction.
- Text detection.
- Model inference.
- Field parsing.
- Total processing duration.

Evaluate:

- CPU thread configuration.
- Batch recognition.
- Reduced unnecessary text regions.
- Image resolution.
- Temporary-file overhead.
- Model warm-up.
- ONNX INT8 inference.
- GPU inference if available.

Make CPU thread settings configurable.

Avoid CPU oversubscription.

Do not claim performance improvements without benchmark results.

Do not sacrifice recognition accuracy solely to reduce latency.

## 13. Phase 8 — Add ONNX model support

Evaluate:

https://huggingface.co/muyleanging/Chakkhu-OCR-Khmer-ONNX

Implement a separate provider:

```text
app/infrastructure/ocr/providers/chakkhu_onnx.py
```

Requirements:

- Implement the existing OCRProvider interface.
- Inspect the published ONNX implementation.
- Match the existing response format.
- Support CPU inference.
- Keep dependencies isolated.
- Register the provider in the OCR factory.
- Allow switching models through configuration.
- Benchmark accuracy, memory, and latency against PyTorch.

Do not replace the working PyTorch provider until ONNX compatibility and accuracy have been verified.

## 14. Configuration requirements

Extend `.env.example` and typed Pydantic Settings.

Suggested configuration:

```ini
APP_ENV=development

OCR_PROVIDER=chakkhu
OCR_DEVICE=cpu
OCR_MIN_CONFIDENCE=0.80
OCR_BATCH_SIZE=8
OCR_CPU_THREADS=4

MODEL_DIR=models

CARD_DETECTOR=opencv
TEXT_DETECTOR=roi
IMAGE_PROCESSOR=original

DOCUMENT_TEMPLATE=ais_student_card_v1

OCR_ENABLE_PERSPECTIVE=true
OCR_ENABLE_ROTATION=true
OCR_DEBUG=false

MAX_FILE_SIZE=10485760
MAX_IMAGE_PIXELS=20000000
```

These are proposed configuration values.

Only expose settings that are implemented, validated, and supported by the selected provider.

Provide sensible defaults and fail-fast configuration validation.

## 15. API response requirements

Preserve existing response compatibility.

Extend the response with optional structured fields and diagnostics.

Example using synthetic data:

```json
{
  "success": true,
  "data": {
    "provider": "chakkhu",
    "document_type": "student_id_card",
    "template": "ais_student_card_v1",
    "confidence": 0.95,
    "processing_time_ms": 1500,
    "fields": {
      "education_level": {
        "raw_text": "បេប័មសិក្សា",
        "value": "បឋមសិក្សា",
        "code": "primary",
        "confidence": 0.89,
        "source": "dictionary_suggestion",
        "requires_review": true
      },
      "student_id": {
        "raw_text": "TEST00001",
        "value": "TEST00001",
        "confidence": 0.98,
        "requires_review": false
      }
    },
    "requires_review": true,
    "timings_ms": {
      "preprocessing": 20,
      "card_detection": 50,
      "text_detection": 30,
      "recognition": 1380,
      "parsing": 20
    },
    "blocks": []
  }
}
```

All example confidence scores and timings are illustrative.

Maintain consistent bounding-box coordinates and document whether they refer to the original, resized, or rectified image.

## 16. Testing and benchmarking

Create:

```text
scripts/
├── test_recognition.py
├── benchmark.py
├── evaluate_accuracy.py
└── visualize_detections.py

tests/
├── unit/
├── integration/
└── fixtures/
```

Implement:

- Unit tests using mock OCR providers.
- Configuration validation tests.
- Unicode normalization tests.
- Vertical text rotation tests.
- ROI coordinate tests.
- Perspective transformation tests.
- Field validation tests.
- API upload validation tests.
- Provider switching tests.
- Empty recognition tests.

Benchmark with manually verified ground-truth text.

Measure:

- Character Error Rate (CER).
- Word Error Rate (WER), with documented Khmer tokenization.
- Field exact-match accuracy.
- Student ID exact-match accuracy.
- Khmer name exact-match accuracy.
- Average latency.
- P50, P95, and P99 latency where sample size supports them.
- Memory utilization.

Compare original RGB, grayscale, CLAHE, ROI detection, and ONNX inference.

Keep a fixed evaluation dataset separate from tuning samples.

Never use model confidence as a substitute for actual accuracy.

## 17. Optional development web interface

After the core OCR pipeline is stable, implement a lightweight Gradio or FastAPI development interface.

Features:

- Image upload.
- Original image preview.
- Rectified card preview.
- Bounding-box visualization.
- Individual ROI previews.
- Khmer and English OCR results.
- Raw and normalized text comparison.
- Confidence scores.
- Model selection for controlled benchmarking.
- Processing-time breakdown.
- JSON response preview.

Reuse existing application services.

Do not duplicate model inference logic in the UI.

Do not expose the debug interface publicly by default.

## 18. Security and privacy

Student ID cards contain sensitive personal information, including information about children.

Requirements:

- Process images locally.
- Do not upload real cards to external OCR services.
- Do not commit real student images to Git.
- Do not log personal names or student IDs.
- Disable image persistence by default.
- Use secure temporary-file handling.
- Validate image types and pixel limits.
- Add authentication before external deployment.
- Protect debug images and benchmark datasets.
- Review downloaded model source code.
- Pin verified model revisions.
- Review model licenses before production deployment.

Use synthetic fixtures in automated tests.

## 19. Development workflow

Maintain the Makefile and add useful targets:

```text
make install
make download-model
make check
make format
make test
make run
make dev
make benchmark
make evaluate
```

Use the existing `.venv` environment.

Do not reinstall large dependencies unnecessarily.

Use four spaces for Python indentation and tabs only for Makefile recipes.

Run syntax validation, import validation, and unit tests after every implementation phase.

## 20. Agent execution rules

1. Inspect the existing repository first.
2. Identify missing files and architectural inconsistencies.
3. Produce a concise implementation plan.
4. Establish baseline measurements.
5. Implement improvements incrementally.
6. Preserve the working Chakkhu integration.
7. Run relevant tests after every phase.
8. Fix encountered errors before continuing.
9. Avoid unrelated refactoring.
10. Do not overwrite user changes unnecessarily.
11. Do not download large models without a clear requirement.
12. Do not train or fine-tune a model until preprocessing and segmentation have been evaluated.
13. Document all assumptions and limitations.
14. Report actual test results, not invented results.

## 21. Expected deliverables

- Improved modular source code.
- Updated project structure.
- Configurable image preprocessing.
- Physical card detection.
- Template-based ROI extraction.
- Vertical Khmer text recognition.
- Structured student ID card parser.
- Interchangeable OCR providers.
- Automated tests.
- Benchmark scripts.
- Updated `.env.example`.
- Updated Makefile.
- Updated README.md.
- Accuracy and performance reports.

## 22. Initial execution priority

Start with these tasks:

1. Audit the current project.
2. Verify the uploaded image using its dimensions and checksum.
3. Establish an OCR accuracy baseline using the original card.
4. Implement ROI extraction.
5. Correct vertical education-level text orientation.
6. Compare Khmer preprocessing strategies.
7. Implement structured field parsing.
8. Optimize recognition performance.
9. Evaluate ONNX as an alternative provider.

Complete and verify these stages before considering custom model training.

Begin by inspecting the current repository and proceed with implementation.
