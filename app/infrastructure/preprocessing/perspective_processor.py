import cv2
import numpy as np
from PIL import Image


def ordered_corners(points: np.ndarray) -> np.ndarray:
    """Return convex quad in TL, TR, BR, BL order without duplicate vertices."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0)
    if cv2.contourArea(ordered, oriented=True) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    if len(np.unique(ordered, axis=0)) != 4 or abs(cv2.contourArea(ordered)) < 16:
        raise ValueError("Degenerate card quadrilateral")
    return ordered


def rectify(image: Image.Image, points: np.ndarray) -> Image.Image:
    tl, tr, br, bl = ordered_corners(points)
    width = max(2, round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = max(2, round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), target)
    return Image.fromarray(
        cv2.warpPerspective(np.asarray(image.convert("RGB")), transform, (width, height))
    )
