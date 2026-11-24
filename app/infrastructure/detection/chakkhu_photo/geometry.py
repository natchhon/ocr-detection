"""Adapted crop grouping and voting from Chakkhu by Ing Muyleang (Apache-2.0).

Source: muyleanging/Chakkhu-OCR-Demo, scene.py blob
0dc7cc83d55957cf36bb5a19ef7378a51e5f2e16. See THIRD_PARTY_NOTICES.md.
Local changes: detector ownership moved to PhotoTextDetector; conservative row
grouping keeps logos, vertical labels and neighboring lines separate; formatting.
"""

import cv2
import numpy as np

PADS = [(0.0, 0.05, 0.15), (0.12, 0.18, 0.25), (0.25, 0.3, 0.3)]


def group(quads) -> list[dict]:
    """Detector quads → text rows, top to bottom."""
    items = []
    for q in quads:
        q = np.asarray(q, dtype=np.float32)
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            items.append((q, [x0, y0, x1, y1]))
    items.sort(
        key=lambda it: it[1][3] - it[1][1], reverse=True
    )  # tall pieces first: they define the rows
    rows = []
    for q, (x0, y0, x1, y1) in items:
        h, cy = y1 - y0, (y0 + y1) / 2
        for r in rows:
            bx, rh = r["box"], r["h"]
            # A tall logo/vertical label must never absorb all nearby horizontal
            # rows. Require similar baselines, heights and a short inter-word gap.
            horizontal = x1 - x0 >= 1.5 * h and bx[2] - bx[0] >= 1.5 * rh
            same = (
                horizontal
                and max(h, rh) <= 1.5 * min(h, rh)
                and abs(cy - (bx[1] + bx[3]) / 2) < 0.35 * min(h, rh)
                and max(x0 - bx[2], bx[0] - x1) < 1.5 * min(h, rh)
            )
            mark = (
                bx[2] - bx[0] >= 1.5 * rh
                and x1 - x0 < 0.7 * rh
                and h < 0.7 * rh
                and bx[1] - 0.2 * rh < cy < bx[3] + 0.2 * rh
                and bx[0] <= x0
                and x1 <= bx[2]
            )
            if same or mark:
                r["quads"].append(q)
                r["box"] = [min(bx[0], x0), min(bx[1], y0), max(bx[2], x1), max(bx[3], y1)]
                break
        else:
            rows.append({"quads": [q], "box": [x0, y0, x1, y1], "h": h})
    return sorted(rows, key=lambda r: (r["box"][1] + r["box"][3]) / 2)


def cut_straight(a: np.ndarray, row: dict, pad, s: float):
    x0, y0, x1, y1 = [v * s for v in row["box"]]
    h = y1 - y0
    H, W = a.shape[:2]
    b = (
        max(0, int(x0 - pad[2] * h)),
        max(0, int(y0 - pad[0] * h)),
        min(W, int(x1 + pad[2] * h)),
        min(H, int(y1 + pad[1] * h)),
    )
    poly = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    return a[b[1] : b[3], b[0] : b[2]], poly


def cut_tilted(a: np.ndarray, row: dict, pad, s: float):
    pts = np.concatenate(row["quads"]) * s
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        w, h, ang = h, w, ang + 90
    if ang > 45:
        ang -= 180
    if ang < -45:
        ang += 180
    if abs(ang) > 30:  # (nearly) vertical text: not handled yet, keep it straight
        ang = 0
    W2, H2 = w + 2 * pad[2] * h, h * (1 + pad[0] + pad[1])
    cy += (pad[1] - pad[0]) * h / 2 * np.cos(np.radians(ang))
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += W2 / 2 - cx
    M[1, 2] += H2 / 2 - cy
    out = cv2.warpAffine(
        a,
        M,
        (max(8, int(W2)), max(8, int(H2))),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    poly = [tuple(map(float, p)) for p in cv2.boxPoints(((cx, cy), (W2, H2), ang))]
    return out, poly


def vote(cands):
    """Combine several readings of one row word by word. cands: [(text, confidence)].
    The most common word count wins; for each word place, the word with the most confidence behind it wins
    (each different crop gets different letters wrong, so the majority is usually right). Returns (text, conf)."""
    from collections import Counter, defaultdict

    cands = [(t.strip(), c) for t, c in cands if t.strip()]
    if not cands:
        return "", 0.0
    n = Counter(len(t.split()) for t, _ in cands).most_common(1)[0][0]
    same = [(t.split(), c) for t, c in cands if len(t.split()) == n]
    if len(same) < 2:
        return max(cands, key=lambda x: x[1])
    words = []
    for i in range(n):
        score = defaultdict(float)
        for w, c in same:
            score[w[i]] += c
        words.append(max(score, key=score.get))
    text = " ".join(words)
    agree = [c for t, c in cands if t == text]
    return text, (max(agree) if agree else sum(c for _, c in same) / len(same))
