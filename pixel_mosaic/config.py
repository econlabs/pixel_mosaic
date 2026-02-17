"""Centralised configuration via a frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MosaicConfig:
    """All tuneable parameters for a mosaic run.

    Attributes:
        max_side:       Longest side of the downscaled image (aspect ratio preserved).
        seed:           Random seed for palette generation (None = non-deterministic).
        share_palette:  Reuse the same palette across all images in a batch.
        palette_mode:   Palette colour theme (see palette.PALETTE_MODES).
        color_space:    Distance metric - "lab" (perceptual) or "rgb".
        solver:         "hungarian" (optimal) or "annealing" (heuristic).
        pixel_upscale:  Each logical pixel becomes n x n in the output image.
        output_format:  Image format for saved files.
        save_palette:   Persist a visualisation of the random palette.
        save_target:    Persist the down-scaled target for comparison.
        save_comparison: Generate a side-by-side comparison grid.
        sa_iterations:  Iteration count for Simulated Annealing solver.
        sa_initial_temp: Starting temperature for SA.
        sa_cooling_rate: Multiplicative cooling factor per iteration for SA.
        sa_gif_frames:  Number of snapshot frames for the SA animation GIF.
        dither:         Apply Floyd-Steinberg error-diffusion dithering.
        input_dir:      Folder to scan for source images.
        output_dir:     Folder for results.
    """

    # Image scaling
    max_side: int = 32  # longest side; short side computed from aspect ratio

    # Palette
    seed: int | None = 42
    share_palette: bool = True
    palette_mode: str = "random"  # see palette.PALETTE_MODES
    palette_source: str | None = None  # path to image to extract palette from

    # Solver
    color_space: str = "lab"
    solver: str = "hungarian"  # "hungarian" | "annealing"

    # Simulated Annealing params
    sa_iterations: int = 2_000_000
    sa_initial_temp: float = 1.0
    sa_cooling_rate: float = 0.999_999
    sa_gif_frames: int = 60

    # Post-processing
    dither: bool = False

    # Output
    pixel_upscale: int = 12
    output_format: str = "png"
    save_palette: bool = True
    save_target: bool = True
    save_comparison: bool = True

    # Paths
    input_dir: Path = field(default_factory=lambda: Path("images"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".jfif"}
    )
