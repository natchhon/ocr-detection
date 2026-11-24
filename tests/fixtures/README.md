Automated tests generate synthetic PIL images in memory or pytest temporary directories.
No real student images or ground truth belong in this directory.

`python -m scripts.create_synthetic_dataset /tmp/ocr-synthetic` generates a fictitious
landscape card, verified text, checksum manifest, and a template calibrated to that
synthetic layout. It is not an AIS template or a held-out accuracy dataset.
Keep authorized evaluation and tuning data in separate private directories.
