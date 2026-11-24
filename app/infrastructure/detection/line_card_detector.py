"""Geometric fallback for card borders interrupted by clips or rounded corners."""

from itertools import combinations

import cv2
import numpy as np


def _deduplicate(lines: list[tuple], span: int, limit: int) -> list[tuple]:
    selected = []
    for line in sorted(lines, reverse=True):
        if all(abs(line[3] - prior[3]) > span * 0.012 for prior in selected):
            selected.append(line)
            if len(selected) == limit:
                break
    return selected


def _intersection(horizontal: tuple, vertical: tuple) -> list[float]:
    _, mh, bh, _ = horizontal
    _, mv, bv, _ = vertical
    y = (mh * bv + bh) / (1 - mh * mv)
    return [mv * y + bv, y]


def find_line_quadrilateral(gray: np.ndarray) -> np.ndarray | None:
    """Return a supported near-upright quadrilateral; never infer card identity.

    Search cost is bounded by downsampling and limiting distinct line hypotheses.
    Thresholds scale with image dimensions; no sample-specific pixel corners.
    """
    original_h, original_w = gray.shape
    scale = min(1.0, 1280 / max(original_h, original_w))
    if scale < 1:
        gray = cv2.resize(gray, (round(original_w * scale), round(original_h * scale)))
    height, width = gray.shape
    short = min(height, width)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(25, round(short / 16)),
        minLineLength=max(30, round(short * 0.1)),
        maxLineGap=max(3, round(short * 0.025)),
    )
    if lines is None:
        return None
    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if abs(dy) < abs(dx) * 0.25:
            slope = dy / dx
            intercept = y1 - slope * x1
            horizontals.append((length, slope, intercept, slope * width / 2 + intercept))
        elif abs(dx) < abs(dy) * 0.25:
            slope = dx / dy
            intercept = x1 - slope * y1
            verticals.append((length, slope, intercept, slope * height / 2 + intercept))
    horizontals = _deduplicate(horizontals, height, 24)
    verticals = _deduplicate(verticals, width, 16)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    tolerance = max(2, short / 240)
    best_score, best_quad = 0.0, None
    for ha, hb in combinations(horizontals, 2):
        top, bottom = sorted((ha, hb), key=lambda line: line[3])
        if bottom[3] - top[3] < height * 0.3:
            continue
        for va, vb in combinations(verticals, 2):
            left, right = sorted((va, vb), key=lambda line: line[3])
            if right[3] - left[3] < width * 0.3:
                continue
            quad = np.array(
                [
                    _intersection(top, left),
                    _intersection(top, right),
                    _intersection(bottom, right),
                    _intersection(bottom, left),
                ],
                dtype=np.float32,
            )
            if not (
                np.all(quad[:, 0] >= width * 0.01)
                and np.all(quad[:, 0] < width * 0.99)
                and np.all(quad[:, 1] >= height * 0.01)
                and np.all(quad[:, 1] < height * 0.99)
            ):
                continue
            area = cv2.contourArea(quad) / (height * width)
            if not 0.25 <= area <= 0.95 or not cv2.isContourConvex(quad):
                continue
            lengths = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
            if min(lengths) < 30 or max(lengths) / min(lengths) > 2.3:
                continue
            support = []
            for i in range(4):
                samples = np.linspace(quad[i], quad[(i + 1) % 4], 150)
                xs = np.clip(samples[:, 0].astype(int), 0, width - 1)
                ys = np.clip(samples[:, 1].astype(int), 0, height - 1)
                support.append(float(np.mean(distance[ys, xs] < tolerance)))
            average = float(np.mean(support))
            if min(support) < 0.45 or average < 0.65:
                continue
            # Prefer the large supported outer boundary over internal text/photo edges.
            score = area * average
            if score > best_score:
                best_score, best_quad = score, quad
    if best_quad is None:
        return None
    best_quad[:, 0] *= original_w / width
    best_quad[:, 1] *= original_h / height
    return best_quad
