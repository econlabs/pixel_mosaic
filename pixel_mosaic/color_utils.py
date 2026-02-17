"""Colour-space conversion and cost-matrix computation."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert flat (N, 3) uint8 RGB → (N, 3) float64 CIELAB."""
    return rgb2lab(rgb.astype(np.float64).reshape(1, -1, 3) / 255.0).reshape(-1, 3)


def compute_cost_matrix(
    palette: np.ndarray,
    target: np.ndarray,
    color_space: str = "lab",
    chunk_size: int = 512,
) -> np.ndarray:
    """Pairwise Euclidean distance between palette and target colours.

    Args:
        palette: (N, 3) uint8 RGB.
        target:  (N, 3) uint8 RGB.
        color_space: ``"lab"`` or ``"rgb"``.
        chunk_size: Rows computed per batch (controls peak RAM).

    Returns:
        (N, N) float32 cost matrix.
    """
    if color_space == "lab":
        p = rgb_to_lab(palette)
        t = rgb_to_lab(target)
    else:
        p = palette.astype(np.float32)
        t = target.astype(np.float32)

    n = len(p)
    cost = np.empty((n, n), dtype=np.float32)
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        diff = p[i:j, np.newaxis, :] - t[np.newaxis, :, :]
        cost[i:j] = np.sqrt(np.sum(diff ** 2, axis=2))
    return cost
