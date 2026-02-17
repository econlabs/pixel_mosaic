"""Floyd-Steinberg error-diffusion dithering.

This is a *post-processing* step applied **after** the assignment.  It
re-orders palette colours along the grid so that quantisation error is
diffused to neighbouring pixels, giving the illusion of smoother tonal
gradients even with a limited colour set.

Note: because the palette is fixed (no new colours are created), dithering
here works by *swapping* assigned colours between nearby positions when
doing so reduces the propagated error.  The result still uses exactly the
same 4 096 colours — only their positions change.
"""

from __future__ import annotations

import numpy as np


def apply_dithering(
    mosaic: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Apply Floyd-Steinberg-style error diffusion via local swaps.

    Instead of modifying pixel values (impossible with a fixed palette),
    we scan the grid and propagate quantisation error to neighbours by
    preferentially swapping colours with nearby pixels that would better
    absorb the accumulated error.

    Args:
        mosaic:  (H, W, 3) uint8 - current mosaic.
        target:  (H, W, 3) uint8 - the down-scaled target.

    Returns:
        (H, W, 3) uint8 - refined mosaic (same set of colours).
    """
    h, w = mosaic.shape[:2]
    result = mosaic.astype(np.float64).copy()
    tgt = target.astype(np.float64)

    error = np.zeros((h, w, 3), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            desired = tgt[y, x] + error[y, x]
            actual = result[y, x]
            quant_err = desired - actual

            # Try swapping with the right or bottom neighbour if it reduces error
            best_swap = None
            best_improvement = 0.0

            for dy, dx in [(0, 1), (1, 0), (1, 1)]:
                ny, nx = y + dy, x + dx
                if ny >= h or nx >= w:
                    continue
                neighbour = result[ny, nx]
                fallback = tgt[min(ny, h - 1), min(nx, w - 1)]
                neighbour_desired = (
                    tgt[ny, nx] + error[ny, nx] if ny < h and nx < w else fallback
                )

                # Current cost
                current = (np.sum((desired - actual) ** 2) +
                          np.sum((neighbour_desired - neighbour) ** 2))
                # Cost after swap
                swapped = (np.sum((desired - neighbour) ** 2) +
                          np.sum((neighbour_desired - actual) ** 2))

                improvement = current - swapped
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (ny, nx)

            if best_swap is not None:
                ny, nx = best_swap
                result[y, x], result[ny, nx] = result[ny, nx].copy(), result[y, x].copy()
                actual = result[y, x]
                quant_err = desired - actual

            # Diffuse error (Floyd-Steinberg weights)
            if x + 1 < w:
                error[y, x + 1] += quant_err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    error[y + 1, x - 1] += quant_err * 3 / 16
                error[y + 1, x] += quant_err * 5 / 16
                if x + 1 < w:
                    error[y + 1, x + 1] += quant_err * 1 / 16

    return np.clip(result, 0, 255).astype(np.uint8)
