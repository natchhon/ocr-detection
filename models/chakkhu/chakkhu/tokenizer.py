"""Cluster tokenizer: one token = one VISIBLE Khmer letter cluster (ស្ត្រី = 1 token), not one Unicode character.

Why: in a picture, a cluster is one shape. Base consonant + subscripts (្ + consonant) + vowels + signs are drawn
together, and some vowels are drawn on the LEFT but typed AFTER the consonant (កេ). If the model writes whole
clusters, it never has to reorder the pieces itself, and lines get about 2x shorter (fewer steps, fewer mistakes).

Rare clusters that are not in the vocabulary fall back to single characters, so ANY text can still be written.

    tok = ClusterTokenizer.load("models/tokenizer.json")
    tok.encode("ស្ត្រីទៅ cafe")   # [BOS?] ids …   (no BOS/EOS added here; the model adds them)
    tok.decode(ids)
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .khnormal import khnormal

PAD, BOS, EOS, UNK, BLANK = 0, 1, 2, 3, 4  # BLANK = CTC "no letter here"
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "<blank>"]

# A Khmer cluster: a base (consonant U+1780–17A2 or independent vowel U+17A3–17B3), then any number of
# subscripts (COENG U+17D2 + consonant) and dependent vowels / signs (U+17B6–17D1, U+17D3, U+17DD).
# Zero-width joiners inside are kept. Anything else (Latin letter, digit, space, punctuation) = one token.
CLUSTER = re.compile(r"[ក-ឳ](?:្[ក-ឳ]|[ា-៑៓៝‌‍])*|.", re.S)


def normalize(text: str) -> str:
    """Khmer normalization (khnormal: one fixed order inside each cluster, invisible characters removed,
    split vowels joined), spaces collapsed."""
    return " ".join(khnormal(text).split())


def clusters(text: str) -> list[str]:
    return CLUSTER.findall(normalize(text))


class ClusterTokenizer:
    def __init__(self, vocab: list[str]):
        self.itos = vocab
        self.stoi = {s: i for i, s in enumerate(vocab)}

    def __len__(self):
        return len(self.itos)

    # ---- build ---------------------------------------------------------------------------------
    @classmethod
    def build(cls, texts, max_clusters: int = 4000, min_count: int = 5) -> "ClusterTokenizer":
        """Vocabulary = specials + every single character seen (the fallback) + the most frequent clusters."""
        cl, ch = Counter(), Counter()
        for t in texts:
            for c in clusters(t):
                cl[c] += 1
                ch.update(c)
        chars = sorted(ch)  # every character seen, so nothing becomes <unk>
        multi = [c for c, n in cl.most_common() if len(c) > 1 and n >= min_count][:max_clusters]
        return cls(SPECIALS + chars + multi)

    # ---- use -----------------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = []
        for c in clusters(text):
            if c in self.stoi:
                ids.append(self.stoi[c])
            else:  # rare cluster: write it character by character
                ids.extend(self.stoi.get(ch, UNK) for ch in c)
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i >= len(SPECIALS) and i < len(self.itos))

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "khmer-cluster", "vocab": self.itos}, ensure_ascii=False, indent=0),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ClusterTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["vocab"])
