# AIS original-card calibration

This follow-up uses the original image subsequently supplied by the user, processed
locally. It supersedes the earlier report's statement that no original was available.
The pasted OCR text includes browser/demo controls that are absent from this JPEG;
that pasted output cannot serve as an original-image benchmark.

## Input and diagnosis

The authorized file is 960×1280, JPEG, 95,030 bytes. Its SHA-256 is
`d4b6a44c0ae5d13df8a590559023ee2a632b4085d4a9cb2636f8e08ea89b472b`.
Images, private OCR text, transcriptions, and debug crops remain in ignored local
`samples/` and `private-data/` directories; no student images or personal field
values are included in this report, application code, or automated fixtures.

The generic baseline on the original produced 25 unnamed blocks in one 2,500.58 ms
service call. The original quadrilateral-contour detector failed to find the card.
Without rectification, the AIS template could not be used, and generic contours
could not reliably segment the vertical education label and small headings.

## Changes and calibration

A bounded line-based fallback now detects supported outer edges when continuous
contours fail. It searches near-upright line pairs, checks in-frame convex geometry,
size/aspect ratio and support along every edge, and favors the supported outer
boundary. Hypotheses are capped and images are downsampled for the search. No
sample-specific pixel corners are embedded in production code. The fallback is
geometric only, has no calibrated confidence probability, and still reports the
existing document-identity warning.

The detected card was rectified to 712×1131. Nine ROIs were measured relative to
that rectified image and stored as normalized coordinates in
`configs/templates/ais_student_card.json`. They exclude the portrait, clip, large
logo, cartoon, and decorative graphics. The academic year is intentionally read
from the lower-right badge. The student-ID region excludes its printed `ID:` label.

Both education-line rotations were inspected. **Clockwise (-90°)** makes the text
upright; counterclockwise leaves it upside down. This is now the template setting.
The template's `calibrated: true` means its coordinates were measured on this one
sample. It does not mean the layout or recognition was validated across AIS cards.

English school names now have a configurable `SCHOOL_NAMES_EN` controlled list.
A mismatched high-confidence school prediction returns a separate dictionary
suggestion and requires review; its original `value` and `raw_text` remain intact.
Personal names now always require human review, without dictionary substitution.
Identifier format validation remains unset until actual business formats are
confirmed; neither ID nor card-number lookalike characters are silently replaced.

## Results and remaining errors

Seven legible fields were transcribed from the image for this tuning check. The
Khmer personal-name spelling and ambiguous card-number prefix were left unverified
and excluded from exact-match scoring pending user confirmation.

| Field | Raw OCR matches visual transcription? |
| --- | --- |
| Khmer school name | Yes |
| English school name | No; two character errors remain |
| Education level | Yes |
| English personal name | Yes; still requires human review |
| Student ID | Yes; no automatic O/0 substitution |
| Campus | Yes |
| Academic year | Yes |
| Khmer personal name | Unverified; no correctness claim |
| Card number | Unverified; no correctness claim |

The Khmer heading reads `សាលារៀន អន្តរទ្វីប អាមេរិកាំង`; the education label reads
`បឋមសិក្សា`. The small English heading is still recognized as
`Americon Iatercon School`, despite high model confidence. Its separate suggested
value is `American Intercon School`. This suggestion is not counted as a raw OCR
match. The raw result therefore matches **6/7 checked fields**, not all nine fields.

On the same rectified ROIs, RGB, grayscale, CLAHE, mild denoising, four-pixel padding,
2× upscaling, and a green-channel experiment all matched the same six checked
fields. CTC-only PyTorch and ONNX beam-4 also left the English heading incorrect.
RGB and the working PyTorch beam-4 decoder remain the measured configuration; no
additional enhancement or model replacement was justified by this sample.

The final service pipeline averaged **2,008.65 ms** over three timed repeats after
one warmup (1,921.26 / 2,034.44 / 2,070.25 ms), excluding startup and upload decoding.
It returns nine named blocks. This is a single tuning sample, not an independent
held-out evaluation. The one-call baseline has different warmup conditions, so no
statistically established speedup is claimed. See
[aggregate measurements](ais-calibration-metrics.json). Full text remains only in
`private-data/ais-result.json`.

## Reproduce and verify

```bash
make recognize-ais
make run-ais
# Upload the original JPEG to the AIS-configured server:
curl -F 'file=@samples/ais_id_card.jpg' http://localhost:8000/api/v1/id-card/detect
```

`make run-ais` explicitly selects RGB, PyTorch beam 4, physical card detection, and
the AIS template. Generic `make run` still defaults to contour OCR. No existing
`.env`, running service, or production infrastructure was changed.

`make check` passes; **60 tests passed in 1.08 seconds**, including synthetic
interrupted-border detection at two resolutions, rejection of insufficient edges,
ROI configuration, both rotation signs, unchanged personal text, review of
high-confidence school-name errors, and existing API/lifecycle regressions. Real
student images are not used as automated test fixtures. Broader layouts, harder
perspective/occlusion, and field accuracy still need independent authorized samples.

## Subsequent fallback diagnostics fix

The AIS launch target now disables generic OCR fallback with
`OCR_ROI_FALLBACK=false`. Failed card/geometry matching returns HTTP 422 before
inference, including input dimensions and checksum. General configuration retains
the previous optional fallback behavior. `/health` exposes the effective detector,
template ID, and fallback flag. This does not improve recognition accuracy or
validate school identity. The supplied Aii Language Center layout is different and
is not supported by this AIS template. The later pasted fragmented output cannot
be reproduced without identifying its input file.
