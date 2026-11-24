import hashlib
import json

import pytest
from PIL import Image

from scripts.benchmark import latency_metrics, load_dataset, write_report
from scripts.evaluate_accuracy import accuracy_metrics


def test_cer_wer_and_exact_fields_have_separate_meanings():
    reference = [{"text": "កខ A", "fields": {"student_id": "TEST00001", "name_kh": "កខ"}}]
    prediction = [
        {
            "text": "កគ A",
            "fields": {"student_id": {"value": "TESTOOO01"}, "name_kh": {"value": "កខ"}},
        }
    ]
    result = accuracy_metrics(reference, prediction)
    assert result["cer"] == 0.25
    assert result["wer_whitespace"] == 0.5
    assert result["field_exact_match"] == 0.5
    assert result["fields"]["student_id"]["exact_match"] == 0
    assert result["fields"]["name_kh"]["exact_match"] == 1


def test_accuracy_nfc_empty_reference_and_cardinality():
    assert accuracy_metrics([{"text": "é"}], [{"text": "e\u0301"}])["cer"] == 0
    blank = accuracy_metrics([{"text": ""}], [{"text": "unexpected"}])
    assert blank["cer"] is None and blank["character_errors"] == 10
    with pytest.raises(ValueError):
        accuracy_metrics([{"text": "x"}], [])


def test_small_samples_do_not_claim_tail_latency():
    assert latency_metrics([1, 2, 3])["p50_ms"] is None
    assert latency_metrics(list(range(5)))["p50_ms"] == 2
    assert latency_metrics(list(range(99)))["p95_ms"] is None
    assert latency_metrics(list(range(100)))["p95_ms"] == pytest.approx(94.05)
    assert latency_metrics(list(range(999)))["p99_ms"] is None


def test_manifest_checksums_splits_and_duplicates(tmp_path):
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (40, 30), "white").save(image)
    sample = {
        "image": image.name,
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "text": "synthetic",
    }
    manifest = tmp_path / "manifest.json"

    def save(split="evaluation", samples=None):
        manifest.write_text(json.dumps({"split": split, "samples": samples or [sample]}))

    save()
    assert len(load_dataset(manifest, "evaluation")[0]) == 1
    save("tuning")
    with pytest.raises(ValueError, match="split"):
        load_dataset(manifest, "evaluation")
    save(samples=[sample, sample])
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(manifest)
    sample["sha256"] = "wrong"
    save()
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(manifest)


def test_private_report(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, {"metric": 0.5})
    assert path.stat().st_mode & 0o777 == 0o600
