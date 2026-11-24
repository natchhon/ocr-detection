# Correction: recognize varying card layouts

The failing response was matched by SHA-256 to `samples/ais_card_v2.jpg`. The old
pipeline chose an interior rectangle as the card and applied nine fixed regions
measured from another image. It returned only four nonempty fragments, assigned
AIS field names without validating their contents, and missed most visible text.

The correction uses the Chakkhu demo's local Photo-mode approach: PP-OCRv4 text
detection (`rapidocr-onnxruntime==1.4.4`), three crop margins, straight/deskewed
variants, and confidence-weighted word voting with the existing Chakkhu reader.
There is no card-boundary crop or mandatory template. The demo source was inspected
locally; no sample images were uploaded. Attribution and license are in
`THIRD_PARTY_NOTICES.md` and `licenses/Apache-2.0.txt`.

Directly copying the demo's row grouping was insufficient: tall vertical labels
absorbed horizontal rows, and logos merged with headings/identifiers. The adapted
grouping requires compatible baselines/heights and shorter gaps. Vertical rows
are isolated, rotated both directions, voted within each direction, and selected
by recognition score. Synthetic regressions cover these failure modes without
embedding any real card's coordinates or personal values.

## Local checks

All five original JPEGs were processed locally with unchanged recognizer weights,
beam 4 and two PyTorch CPU threads. These are single runs, not an independent
accuracy benchmark. Some verification ran concurrently, so latency is descriptive.

| Sample | Returned text blocks | Observed outcome / remaining problems |
| --- | ---: | --- |
| aii_id_card.jpg | 12 | Name, ID line and vertical number recovered; stylized logos and a program-heading character still wrong. |
| ais_card_v2.jpg | 12 | Student-ID digits, campus and education label recovered; English name includes a decorative trailing character; small headings remain inaccurate. |
| ais_card_v3.jpg | 11 | No geometry rejection; name, ID digits and education label recovered; ID punctuation and tiny/occluded headings remain inaccurate. |
| ais_card_v4.jpg | 12 | Name, ID line, campus, year and education label recovered; small heading and logo errors remain. |
| ais_id_card.jpg | 12 | Main text recovered without the fixed template; English heading remains imperfect and Khmer personal-name spelling still needs verification. |

Block counts are not accuracy scores. No comprehensive, independent transcription
set exists for these five images, so no CER or accuracy percentage is claimed.
Private full outputs are in the ignored `private-data/photo-corrected/` directory;
public aggregate measurements are in `photo-pipeline-metrics.json`.

A real FastAPI TestClient request using the effective local configuration and
default ten PyTorch threads processed the failing v2 JPEG successfully: HTTP 200,
12 blocks, correct student-ID digits, full 877×1280 coordinate frame, no template
and no assumed field labels. It took **29,886.77 ms**, versus the supplied old
1,891.94 ms response that read only four incorrect fragments. Multiple crop trials
cost more CPU work. The five two-thread service runs ranged from about 19–40 seconds.

`make check` passed. **74 tests passed**, including grouping, vertical orientation,
coordinate scaling/bounds, voting isolation and raw-text preservation, startup and
shutdown, upload validation, and the absence of forced AIS labels in Photo mode.
FastAPI threaded tests timed out inside the sandbox and passed outside it.

## Run the corrected pipeline

Restart the existing server with `make run-photo`. The old `make run-ais` command
now aliases this mode. `/health` should report `text_detector: "photo"` and
`template: null`. The local `.env` detector was changed from `opencv` to `photo`;
no production infrastructure or running external server was changed.

`text` contains the voted reading; `raw_text` keeps the highest-confidence crop's
reader output in the selected orientation; `candidate_texts` preserves all crop
readings. Named `fields` stay empty in Photo mode. Field extraction needs its own
validated interpretation of the detected text, rather than assigning names based
on one card's positions. High model confidence does not certify an identifier.

This is the demo's Photo approach with the described grouping/vertical changes,
not an implementation of its Auto page/line fallback or Document export. Optional
legacy ROI/contour modes remain explicitly selectable for comparison. Future
changes should be evaluated across all samples and a separate held-out set.
