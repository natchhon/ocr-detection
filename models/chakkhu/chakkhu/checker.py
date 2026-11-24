"""Khmer word checker: a small cluster n-gram language model that knows which Khmer spellings are real.

The reader already writes ~5 guesses per line (decoder beam + CTC draft), each with two image scores.
The checker adds a third score, "how much does this look like real Khmer text", and the best total wins:

    total = λ·CTC + (1−λ)·decoder + μ·(LM + β·clusters)

So the checker can only choose among guesses the picture supports; it never rewrites text on its own.
The LM is built from our own clean training text only (Akkhara sentences, Khmer Wikipedia, sign words).

Case-blind: Latin text is lower-cased before scoring, so CAPS vs lowercase is left to the picture.

Storage: each n-gram of cluster ids is packed into one int64 (13 bits per id, order ≤ 4 + ...).
Counts are sorted arrays, looked up with np.searchsorted, so it is small and fast on the CPU.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .tokenizer import clusters, normalize

BITS = 13  # cluster ids < 8192
BOS_ID, EOS_ID, UNK_ID = 1, 2, 3  # our own ids inside the LM (0 = padding)


class ClusterLM:
    def __init__(self, vocab: dict[str, int], keys: list[np.ndarray], counts: list[np.ndarray], order: int):
        self.vocab, self.keys, self.counts, self.order = vocab, keys, counts, order
        # context totals: for n-gram order n, total count of its (n-1) prefix is stored in the order n-1 table
        self.total1 = float(counts[0].sum())

    # ---------- build ----------
    @classmethod
    def build(cls, lines, order: int = 5, min_vocab: int = 2) -> "ClusterLM":
        from collections import Counter
        cc = Counter()
        seqs = []
        for line in lines:
            cl = clusters(normalize(line.strip()).lower())
            if cl:
                cc.update(cl)
                seqs.append(cl)
        vocab = {c: i + 4 for i, (c, n) in enumerate(cc.most_common()) if n >= min_vocab}
        assert len(vocab) + 4 < (1 << BITS), len(vocab)
        ids = []
        for cl in seqs:
            ids.extend([BOS_ID] + [vocab.get(c, UNK_ID) for c in cl] + [EOS_ID, 0])  # 0 separates lines
        arr = np.array(ids, dtype=np.int64)
        keys, counts = [], []
        for n in range(1, order + 1):
            if len(arr) < n:
                break
            packed = np.zeros(len(arr) - n + 1, dtype=np.int64)
            ok = np.ones(len(arr) - n + 1, dtype=bool)
            for j in range(n):
                part = arr[j: len(arr) - n + 1 + j]
                packed = (packed << BITS) | part
                ok &= part != 0
            k, c = np.unique(packed[ok], return_counts=True)
            keys.append(k)
            counts.append(c.astype(np.int64))
        return cls(vocab, keys, counts, order)

    def save(self, path):
        np.savez_compressed(path, order=self.order, vocab=np.array(list(self.vocab.keys())),
                            vocab_ids=np.array(list(self.vocab.values())),
                            **{f"k{n}": k for n, k in enumerate(self.keys)},
                            **{f"c{n}": c for n, c in enumerate(self.counts)})

    @classmethod
    def load(cls, path) -> "ClusterLM":
        z = np.load(path, allow_pickle=False)
        order = int(z["order"])
        vocab = dict(zip(z["vocab"].tolist(), z["vocab_ids"].tolist()))
        n = sum(1 for f in z.files if f.startswith("k"))
        return cls(vocab, [z[f"k{i}"] for i in range(n)], [z[f"c{i}"] for i in range(n)], order)

    # ---------- score ----------
    def _count(self, gram: tuple[int, ...]) -> int:
        n = len(gram)
        key = 0
        for g in gram:
            key = (key << BITS) | g
        k = self.keys[n - 1]
        i = np.searchsorted(k, key)
        return int(self.counts[n - 1][i]) if i < len(k) and k[i] == key else 0

    def _ids(self, text):
        return [BOS_ID] + [self.vocab.get(c, UNK_ID) for c in clusters(normalize(text).lower())] + [EOS_ID]

    def token_logps(self, ids, alpha: float = 0.4) -> list[float]:
        """Stupid-backoff log probability of each id after the first (natural log)."""
        out = []
        for t in range(1, len(ids)):
            pen = 0.0
            for n in range(min(self.order, t + 1), 0, -1):
                gram = tuple(ids[t - n + 1: t + 1])
                c = self._count(gram)
                if c:
                    ctx = max(c, self._count(gram[:-1]) if n > 1 else self.total1)   # (pruned contexts can count 0)
                    out.append(pen + math.log(c / ctx))
                    break
                pen += math.log(alpha)
            else:
                out.append(pen + math.log(0.5 / self.total1))
        return out

    def logp(self, text: str, alpha: float = 0.4) -> float:
        """Stupid-backoff log probability (natural log) of a whole line, including its end."""
        return sum(self.token_logps(self._ids(text), alpha))

    def next_ids(self, ctx: list[int], top: int = 5) -> list[int]:
        """Most likely next cluster ids after this context (longest context that was seen)."""
        for n in range(min(self.order, len(ctx) + 1), 1, -1):
            pre = 0
            for g in ctx[len(ctx) - n + 1:]:
                pre = (pre << BITS) | g
            k = self.keys[n - 1]
            lo, hi = np.searchsorted(k, pre << BITS), np.searchsorted(k, (pre + 1) << BITS)
            if hi > lo:
                c = self.counts[n - 1][lo:hi]
                best = np.argsort(-c)[: top + 2]
                ids = [int(k[lo + i] & ((1 << BITS) - 1)) for i in best]
                return [i for i in ids if i > UNK_ID][:top]
        return []

    @property
    def inv(self):
        if not hasattr(self, "_inv"):
            self._inv = {i: c for c, i in self.vocab.items()}
        return self._inv


# ---------------- spelling-fix proposals ----------------
LATIN = re.compile(r"[A-Za-z][A-Za-z']*")


def _case_like(word: str, model: str) -> str:
    if model.isupper() and len(model) > 1:
        return word.upper()
    if model[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


class Proposer:
    """Suggests corrected versions of a read line. The reader then scores each one against the picture."""

    def __init__(self, lm: ClusterLM, latin_words: dict[str, int]):
        self.lm = lm
        self.words = latin_words
        self.by_len: dict[int, list[str]] = {}
        for w in latin_words:
            self.by_len.setdefault(len(w), []).append(w)

    def near_words(self, w: str, maxd: int = 2, top: int = 3) -> list[str]:
        from rapidfuzz.distance import Levenshtein
        w = w.lower()
        if w in self.words or len(w) < 3:
            return []
        d = 1 if len(w) <= 4 else maxd
        cands = []
        for L in range(len(w) - d, len(w) + d + 1):
            for v in self.by_len.get(L, ()):
                e = Levenshtein.distance(w, v, score_cutoff=d)
                if e <= d:
                    cands.append((e, -self.words[v], v))
        return [v for _, _, v in sorted(cands)[:top]]

    def latin_fixes(self, text: str) -> list[str]:
        out = []
        for m in LATIN.finditer(text):
            w = m.group()
            for v in self.near_words(w):
                out.append(text[: m.start()] + _case_like(v, w) + text[m.end():])
            if len(w) >= 6 and w.lower() not in self.words:  # stuck-together words: MODARKING → NO PARKING
                for i in range(2, len(w) - 1):
                    a, b = w[:i], w[i:]
                    va = [a.lower()] if a.lower() in self.words else self.near_words(a, 1, 1)
                    vb = [b.lower()] if b.lower() in self.words else self.near_words(b, 1, 1)
                    if va and vb and len(va[0]) >= 2:
                        out.append(text[: m.start()] + _case_like(va[0], a) + " " + _case_like(vb[0], b) + text[m.end():])
        return out

    def khmer_fixes(self, text: str, n_pos: int = 6, top: int = 4) -> list[str]:
        """At the most surprising clusters: swap in the clusters the LM expects, or drop the cluster."""
        cl = clusters(normalize(text))
        if not cl:
            return []
        ids = [BOS_ID] + [self.lm.vocab.get(c.lower(), UNK_ID) for c in cl] + [EOS_ID]
        lps = self.lm.token_logps(ids)  # lps[t] = logp of ids[t+1] = cluster t (last one = end)
        pos = [t for t in sorted(range(len(cl)), key=lambda t: lps[t]) if re.match(r"[ក-៿]", cl[t])][:n_pos]
        out = []
        for t in pos:
            for j in self.lm.next_ids(ids[: t + 1], top):
                c = self.lm.inv[j]
                if c != cl[t].lower():
                    out.append("".join(cl[:t] + [c] + cl[t + 1:]))      # replace
                    out.append("".join(cl[:t] + [c] + cl[t:]))          # insert before
            out.append("".join(cl[:t] + cl[t + 1:]))                    # drop
        return out

    def propose(self, text: str, limit: int = 60) -> list[str]:
        seen, out = {text}, []
        for v in self.latin_fixes(text) + self.khmer_fixes(text):
            v = normalize(v)
            if v not in seen and v.strip():
                seen.add(v)
                out.append(v)
        return out[:limit]


_LEADER = re.compile(r"([.·…_\-])\1{2,}")


def _leaders(text: str) -> str:
    """'ឈ្មោះ.......អាយុ' -> 'ឈ្មោះ...អាយុ': any run of 3+ same dots/dashes/underscores becomes 3."""
    return _LEADER.sub(r"\1\1\1", text)


def _words(text: str) -> int:
    """Clusters that are letters or digits (not spaces, dots or other marks)."""
    return sum(1 for c in clusters(text) if c[0].isalnum())


def latin_words(root: Path, lm_text_files=None) -> dict[str, int]:
    """English / sign word counts: our word lists + Latin words seen ≥3 times in the training text."""
    from collections import Counter
    words = Counter()
    for l in (root / "hf/data/text/english_words.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        w, _, c = l.partition("\t")
        if w.isascii() and w.isalpha():
            words[w.lower()] += int(c or 1)
    for l in (root / "hf/data/text/sign_words.txt").read_text(encoding="utf-8").splitlines():
        for w in LATIN.findall(l):
            words[w.lower()] += 1000
    seen = Counter()
    for f in lm_text_files or []:
        for l in Path(f).read_text(encoding="utf-8").splitlines():
            seen.update(w.lower() for w in LATIN.findall(l))
    for w, c in seen.items():
        if c >= 3:
            words[w] += c
    return dict(words)


def pick(guesses, lm: ClusterLM, lam: float = 0.5, mu: float = 0.0, beta: float = 0.0):
    """guesses: [(text, decoder_logp, ctc_logp)]. Returns the chosen text."""
    def total(g):
        s = lam * g[2] + (1 - lam) * g[1]
        if mu:
            s += mu * (lm.logp(g[0]) + beta * len(clusters(g[0])))
        return s
    return max(guesses, key=total)[0]


def default_text_files(root: Path):
    return [root / "hf/data/text/khmer_sentences.txt", root / "hf/data/text/khmer_wiki.txt",
            root / "hf/data/text/sign_words.txt"]


class Checker:
    """The whole word checker for live reading: pick + spelling-fix proposals scored against the picture.

    Settings come from the fair KHOB test (tuned on one half, scored on the other): μ .3, β 6, γ 1 per proposal.
    """

    def __init__(self, lm: ClusterLM, words: dict[str, int], mu=.3, beta=6., gamma=1., lam=.5, rounds=2):
        self.lm, self.prop = lm, Proposer(lm, words)
        self.mu, self.beta, self.gamma, self.lam, self.rounds = mu, beta, gamma, lam, rounds
        self.changed: set[str] = set()   # texts the checker chose over the reader's own pick

    @classmethod
    def load(cls, folder, **kw) -> "Checker":
        import json
        folder = Path(folder)
        return cls(ClusterLM.load(folder / "checker_lm.npz"),
                   json.loads((folder / "latin_words.json").read_text(encoding="utf-8")), **kw)

    def total(self, g):
        """g = (text, decoder_logp, ctc_logp, is_proposal)

        The LM and the length bonus see dot leaders ("......") as ONE mark: otherwise every extra dot earns the bonus
        almost for free and a form line "ឈ្មោះ......អាយុ......" loses its words to a guess made only of dots."""
        t = _leaders(g[0])
        return (self.lam * g[2] + (1 - self.lam) * g[1]
                + self.mu * (self.lm.logp(t) + self.beta * len(clusters(t))) - self.gamma * g[3])

    def choose(self, tok, model, memory, mem_pad, ctc_lp, lens, scored):
        """scored: per line [(ids, dec, ctc)] from score_guesses. Returns per line the chosen text."""
        from .decode import score_guesses
        cands = [[(normalize(tok.decode(i)), d, c, 0) for i, d, c in s] for s in scored]
        for rnd in range(1, self.rounds + 1):
            want = []
            for cs in cands:
                if rnd == 1:
                    bases = [g[0] for g in sorted(cs, key=lambda g: -(g[1] + g[2]))[:2]]
                else:
                    bases = [max(cs, key=self.total)[0]]
                have, vs = {g[0] for g in cs}, []
                for b in bases:
                    for v in self.prop.propose(b):
                        if v in have:
                            continue
                        ids = tok.encode(v)
                        if ids and normalize(tok.decode(ids)) == v:
                            have.add(v)
                            vs.append((v, ids))
                want.append(vs)
            if not any(want):
                break
            sc = score_guesses(model, memory, mem_pad, ctc_lp, lens, [[ids for _, ids in vs] for vs in want])
            for cs, vs, s in zip(cands, want, sc):
                cs += [(v, d, c, 1) for (v, _), (_, d, c) in zip(vs, s)]
        out = []
        for cs in cands:
            best = max(cs, key=self.total)[0]
            plain = max((g for g in cs if not g[3]), key=lambda g: self.lam * g[2] + (1 - self.lam) * g[1])[0]
            if _words(best) < 0.8 * _words(plain):   # the checker fixes spellings, it never drops real words
                best = plain
            if best != plain:
                self.changed.add(best)
            out.append(best)
        return out
