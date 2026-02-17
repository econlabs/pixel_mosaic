"""Optimal assignment via the Hungarian algorithm (scipy)."""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from pixel_mosaic.color_utils import compute_cost_matrix

logger = logging.getLogger(__name__)


def solve_hungarian(
    palette: np.ndarray,
    target_flat: np.ndarray,
    color_space: str = "lab",
) -> np.ndarray:
    """Find the optimal palette → position assignment.

    Args:
        palette:     (N, 3) uint8 - the fixed colour set.
        target_flat: (N, 3) uint8 - flattened target pixels (row-major).
        color_space: ``"lab"`` or ``"rgb"``.

    Returns:
        (N, 3) uint8 - the mosaic pixels in row-major order.
    """
    n = len(palette)

    logger.info("Building %dx%d cost matrix (%s) …", n, n, color_space)
    t0 = time.perf_counter()
    cost = compute_cost_matrix(palette, target_flat, color_space)
    logger.info("Cost matrix ready  (%.1f s)", time.perf_counter() - t0)

    logger.info("Running linear_sum_assignment …")
    t0 = time.perf_counter()
    row_idx, col_idx = linear_sum_assignment(cost)
    logger.info("Assignment solved  (%.1f s)", time.perf_counter() - t0)

    mosaic = np.zeros_like(palette)
    for pi, pos in zip(row_idx, col_idx, strict=False):
        mosaic[pos] = palette[pi]
    return mosaic
