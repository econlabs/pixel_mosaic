"""Simulated Annealing solver with optional GIF animation export."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pixel_mosaic.color_utils import rgb_to_lab

logger = logging.getLogger(__name__)


def _pixel_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance between two colour vectors."""
    d = a - b
    return float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)


def solve_annealing(
    palette: np.ndarray,
    target_flat: np.ndarray,
    color_space: str = "lab",
    iterations: int = 2_000_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.999_999,
    gif_frames: int = 60,
    gif_path: Path | None = None,
    img_width: int = 32,
    img_height: int = 32,
    pixel_upscale: int = 12,
) -> np.ndarray:
    """Heuristic solver using Simulated Annealing.

    Args:
        palette:       (N, 3) uint8.
        target_flat:   (N, 3) uint8 (row-major).
        color_space:   "lab" or "rgb".
        iterations:    Total SA iterations.
        initial_temp:  Starting temperature.
        cooling_rate:  Multiplicative cooling per iteration.
        gif_frames:    How many snapshots to capture for the GIF.
        gif_path:      If given, save an animated GIF showing convergence.
        img_width:     Width of the target image.
        img_height:    Height of the target image.
        pixel_upscale: Upscale factor for GIF frames.

    Returns:
        (N, 3) uint8 - the mosaic pixels in row-major order.
    """
    n = len(palette)
    rng = np.random.default_rng()

    # Convert to working colour space
    if color_space == "lab":
        target_cs = rgb_to_lab(target_flat).astype(np.float32)
        palette_cs = rgb_to_lab(palette).astype(np.float32)
    else:
        target_cs = target_flat.astype(np.float32)
        palette_cs = palette.astype(np.float32)

    # Random initial assignment
    logger.info("Building random initial assignment ...")
    assignment = np.arange(n)
    rng.shuffle(assignment)

    mosaic_cs = palette_cs[assignment].copy()

    # Per-pixel cost
    costs = np.sum((mosaic_cs - target_cs) ** 2, axis=1)
    total_cost = float(np.sum(costs))

    logger.info(
        "SA start  | iterations=%s  temp=%.4f  cooling=%.6f",
        f"{iterations:,}", initial_temp, cooling_rate,
    )

    # GIF frame capture
    frames: list[Image.Image] = []
    frame_interval = (
        max(1, iterations // gif_frames) if gif_frames > 0 else iterations + 1
    )

    def _capture_frame() -> None:
        if gif_path is None:
            return
        rgb = palette[assignment]
        img = Image.fromarray(rgb.reshape(img_height, img_width, 3))
        img = img.resize(
            (img_width * pixel_upscale, img_height * pixel_upscale),
            Image.NEAREST,
        )
        frames.append(img)

    _capture_frame()

    temp = initial_temp
    accepted = 0
    t0 = time.perf_counter()

    for it in range(iterations):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue

        old_cost = costs[i] + costs[j]
        new_ci = np.sum((mosaic_cs[j] - target_cs[i]) ** 2)
        new_cj = np.sum((mosaic_cs[i] - target_cs[j]) ** 2)
        new_cost = new_ci + new_cj
        delta = new_cost - old_cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            assignment[i], assignment[j] = assignment[j], assignment[i]
            mosaic_cs[i], mosaic_cs[j] = mosaic_cs[j].copy(), mosaic_cs[i].copy()
            costs[i] = new_ci
            costs[j] = new_cj
            total_cost += delta
            accepted += 1

        temp *= cooling_rate

        if (it + 1) % frame_interval == 0:
            _capture_frame()
            elapsed = time.perf_counter() - t0
            pct = (it + 1) / iterations * 100
            logger.info(
                "  SA %5.1f%%  cost=%.0f  temp=%.2e  accepted=%s  (%.0f s)",
                pct, total_cost, temp, f"{accepted:,}", elapsed,
            )

    _capture_frame()

    elapsed = time.perf_counter() - t0
    logger.info(
        "SA done   | cost=%.0f  accepted=%s/%s  (%.1f s)",
        total_cost, f"{accepted:,}", f"{iterations:,}", elapsed,
    )

    if gif_path is not None and frames:
        gif_path = Path(gif_path)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
        )
        logger.info("SA animation saved: %s (%d frames)", gif_path, len(frames))

    return palette[assignment]
