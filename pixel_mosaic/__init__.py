"""
Pixel Mosaic Generator
======================

Arrange a fixed palette of random colours into a grid that best
approximates any target image. Aspect ratio is preserved.
Ships two solvers:

- **Hungarian** (optimal)
- **Simulated Annealing** (heuristic, tuneable speed / quality trade-off)
"""

__version__ = "1.1.0"

from pixel_mosaic.config import MosaicConfig
from pixel_mosaic.image_io import (
    compute_target_size,
    load_and_resize,
    make_comparison_grid,
    save_upscaled,
)
from pixel_mosaic.palette import (
    extract_palette_from_image,
    generate_palette,
    generate_random_palette,
)
from pixel_mosaic.solver_annealing import solve_annealing
from pixel_mosaic.solver_hungarian import solve_hungarian

__all__ = [
    "MosaicConfig",
    "compute_target_size",
    "extract_palette_from_image",
    "generate_palette",
    "generate_random_palette",
    "load_and_resize",
    "make_comparison_grid",
    "save_upscaled",
    "solve_annealing",
    "solve_hungarian",
]
