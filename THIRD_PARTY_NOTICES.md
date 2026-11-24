# Chakkhu Photo-mode integration

Chakkhu by **Ing Muyleang**, licensed under Apache-2.0.

`app/infrastructure/detection/chakkhu_photo/geometry.py` adapts grouping, crop
generation and voting from the public Chakkhu-OCR-Demo `scene.py`:
https://huggingface.co/spaces/muyleanging/Chakkhu-OCR-Demo/blob/main/scene.py

Inspected Git blob: `0dc7cc83d55957cf36bb5a19ef7378a51e5f2e16`.
The demo repository declares Apache-2.0 in its README metadata.
Changes are described in the adapted module. The license is included in
`licenses/Apache-2.0.txt`.

Photo detection uses RapidOCR ONNX Runtime **1.4.4** and its bundled PP-OCRv4
detector, as used by that demo. RapidOCR and PaddleOCR are Apache-2.0 projects;
their installed distributions retain upstream notices and model files.
The existing local Chakkhu recognizer weights are unchanged.
