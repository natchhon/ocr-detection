"""Chakkhu ចក្ខុ: read Khmer + English text LINES from pictures.

    from chakkhu import load, read
    model, tok = load()                       # downloads muyleanging/Chakkhu-OCR-Khmer (or a local folder)
    read(model, tok, ["line1.png", "line2.jpg"])   # → ["ព្រះរាជាណាចក្រកម្ពុជា", ...]
    read(model, tok, ["line1.png"], confidence=True)   # → [("ព្រះរាជាណាចក្រកម្ពុជា", 0.98)]

One picture = one line of text (cut the line out first; whole pages need a line finder).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

from .checker import Checker
from .decode import decoder_beam, joint_read, score_guesses
from .model import Chakkhu, Config
from .tokenizer import ClusterTokenizer, normalize

HEIGHT = 64
MAX_W = 2400


def load(repo: str = "muyleanging/Chakkhu-OCR-Khmer", device: str = "cpu"):
    """Load the model from a Hugging Face repo id or a local folder. Returns (model, tokenizer)."""
    folder = Path(repo)
    if not folder.exists():
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo))
    tok = ClusterTokenizer.load(folder / "tokenizer.json")
    cfg = Config(**json.loads((folder / "config.json").read_text()))
    model = Chakkhu(cfg)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(folder / "model.safetensors"))
    return model.to(device).eval(), tok


def to_line(img: Image.Image) -> Image.Image:
    """Grey, 64 px high (width keeps the shape)."""
    img = img.convert("L")
    w = max(16, round(img.width * HEIGHT / max(1, img.height)))
    img = img.resize((w, HEIGHT), Image.BILINEAR)
    return img if img.width <= MAX_W else img.resize((MAX_W, HEIGHT), Image.BILINEAR)


@torch.no_grad()
def read(model, tok, images, beam: int = 4, lam: float = 0.5, batch: int = 16, confidence: bool = False,
         checker: Checker | None = None):
    """images: file paths or PIL images, one text line each. Reads with BOTH heads (decoder beam + CTC check).
    beam=0 → fast CTC-only reading.
    confidence=True → list of (text, confidence 0–1). Confidence = how much the two heads agree × how sure the
    CTC columns are. On old printed pages: the most sure half of lines had 0.7% errors, the least sure 10% had 9%.
    checker=Checker.load(folder) → the Khmer word checker picks among the guesses and proposes spelling fixes,
    each scored against the picture (KHOB with full5: 3.90% → 3.58% CER)."""
    dev = next(model.parameters()).device
    lines = [to_line(Image.open(i) if isinstance(i, (str, Path)) else i) for i in images]
    order = sorted(range(len(lines)), key=lambda k: lines[k].width)
    out = [""] * len(lines)
    conf = [0.0] * len(lines)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        part = [lines[k] for k in idx]
        W = (max(im.width for im in part) + 7) // 8 * 8
        x = torch.ones(len(part), 1, HEIGHT, W)
        for j, im in enumerate(part):
            x[j, 0, :, : im.width] = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(HEIGHT, im.width) / 255
        w = torch.tensor([im.width for im in part])
        x, w = x.to(dev), w.to(dev)
        texts = None
        if beam and checker is not None:
            memory, mem_pad, lp, lens = model.encode(x, w)
            drafts = model.ctc_greedy(lp, lens)
            beams = decoder_beam(model, memory, mem_pad, [int(1.5 * len(d)) + 4 for d in drafts], beam)
            guesses = [[list(b) for b in bs] + ([d] if d not in [list(b) for b in bs] else []) for bs, d in zip(beams, drafts)]
            scored = score_guesses(model, memory, mem_pad, lp, lens, guesses)
            texts = checker.choose(tok, model, memory, mem_pad, lp, lens, scored)
            ids = [None] * len(texts)
        elif beam:
            res, drafts = joint_read(model, x, w, k=beam, lams=(lam,))
            ids = res[lam]
        else:
            _, _, lp, lens = model.encode(x, w)
            ids = drafts = model.ctc_greedy(lp, lens)
        if confidence:
            _, _, lp, lens = model.encode(x, w)
            best = lp.exp().max(-1).values
        for j, (k, i) in enumerate(zip(idx, ids)):
            out[k] = texts[j] if texts is not None else normalize(tok.decode(i))
            if confidence:
                d = normalize(tok.decode(drafts[j]))
                agree = 1 - Levenshtein.normalized_distance(out[k], d)
                conf[k] = round(agree * best[j, : lens[j]].mean().item(), 3)
    return list(zip(out, conf)) if confidence else out


__all__ = ["load", "read", "Chakkhu", "Config", "ClusterTokenizer", "Checker"]
