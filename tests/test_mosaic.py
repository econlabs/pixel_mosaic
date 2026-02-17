"""Tests for the pixel_mosaic package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_mosaic.color_utils import compute_cost_matrix, rgb_to_lab
from pixel_mosaic.config import MosaicConfig
from pixel_mosaic.dithering import apply_dithering
from pixel_mosaic.image_io import compute_target_size, load_and_resize, save_upscaled
from pixel_mosaic.palette import (
    extract_palette_from_image,
    generate_palette,
    generate_random_palette,
)
from pixel_mosaic.solver_annealing import solve_annealing
from pixel_mosaic.solver_hungarian import solve_hungarian

# -- Fixtures ----------------------------------------------------------

W, H = 10, 6  # non-square for aspect ratio testing
N = W * H


@pytest.fixture
def palette() -> np.ndarray:
    return generate_random_palette(N, seed=123)


@pytest.fixture
def target() -> np.ndarray:
    """Synthetic non-square target image."""
    rng = np.random.default_rng(456)
    return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


@pytest.fixture
def target_flat(target: np.ndarray) -> np.ndarray:
    return target.reshape(-1, 3)


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    """Write a small non-square test PNG to disk."""
    img = Image.fromarray(
        np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8),
    )
    p = tmp_path / "test.png"
    img.save(p)
    return p


# -- Config ------------------------------------------------------------

class TestConfig:
    def test_defaults(self) -> None:
        cfg = MosaicConfig()
        assert cfg.max_side == 32

    def test_custom_max_side(self) -> None:
        cfg = MosaicConfig(max_side=64)
        assert cfg.max_side == 64

    def test_frozen(self) -> None:
        cfg = MosaicConfig()
        with pytest.raises(AttributeError):
            cfg.max_side = 128  # type: ignore[misc]


# -- Aspect ratio ------------------------------------------------------

class TestAspectRatio:
    def test_landscape(self) -> None:
        w, h = compute_target_size(1920, 1080, 64)
        assert w == 64
        assert h == 36

    def test_portrait(self) -> None:
        w, h = compute_target_size(1080, 1920, 64)
        assert w == 36
        assert h == 64

    def test_square(self) -> None:
        w, h = compute_target_size(500, 500, 32)
        assert w == 32
        assert h == 32

    def test_minimum_one(self) -> None:
        w, h = compute_target_size(1000, 1, 32)
        assert w == 32
        assert h >= 1


# -- Palette -----------------------------------------------------------

class TestPalette:
    def test_shape_and_dtype(self, palette: np.ndarray) -> None:
        assert palette.shape == (N, 3)
        assert palette.dtype == np.uint8

    def test_reproducible(self) -> None:
        a = generate_random_palette(100, seed=42)
        b = generate_random_palette(100, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self) -> None:
        a = generate_random_palette(100, seed=1)
        b = generate_random_palette(100, seed=2)
        assert not np.array_equal(a, b)

    def test_extract_from_image(self, tmp_image: Path) -> None:
        p = extract_palette_from_image(tmp_image, 64, seed=0)
        assert p.shape == (64, 3)
        assert p.dtype == np.uint8

    def test_palette_modes(self) -> None:
        for mode in ["random", "pastel", "grayscale", "warm", "cool",
                      "earth", "neon", "sunset", "ocean", "forest"]:
            p = generate_palette(100, mode=mode, seed=42)
            assert p.shape == (100, 3)
            assert p.dtype == np.uint8

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            generate_palette(100, mode="invalid_mode")


# -- Colour utilities --------------------------------------------------

class TestColorUtils:
    def test_lab_shape(self) -> None:
        rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)
        assert lab.shape == (2, 3)

    def test_cost_matrix_shape(
        self, palette: np.ndarray, target_flat: np.ndarray,
    ) -> None:
        cost = compute_cost_matrix(palette, target_flat, "lab")
        assert cost.shape == (N, N)
        assert cost.dtype == np.float32

    def test_cost_zero_diagonal_identical(self) -> None:
        colors = np.array([[100, 100, 100], [200, 200, 200]], dtype=np.uint8)
        cost = compute_cost_matrix(colors, colors, "rgb")
        np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-5)


# -- Hungarian solver --------------------------------------------------

class TestHungarian:
    def test_uses_all_palette_colours(
        self, palette: np.ndarray, target_flat: np.ndarray,
    ) -> None:
        mosaic = solve_hungarian(palette, target_flat, "lab")
        assert mosaic.shape == palette.shape
        palette_set = {tuple(c) for c in palette}
        mosaic_set = {tuple(c) for c in mosaic}
        assert palette_set == mosaic_set

    def test_no_duplicate_positions(
        self, palette: np.ndarray, target_flat: np.ndarray,
    ) -> None:
        mosaic = solve_hungarian(palette, target_flat, "lab")
        palette_sorted = np.sort(palette, axis=0)
        mosaic_sorted = np.sort(mosaic, axis=0)
        np.testing.assert_array_equal(palette_sorted, mosaic_sorted)

    def test_perfect_match(self) -> None:
        colors = generate_random_palette(16, seed=99)
        mosaic = solve_hungarian(colors, colors.copy(), "rgb")
        np.testing.assert_array_equal(
            np.sort(mosaic, axis=0), np.sort(colors, axis=0),
        )


# -- Simulated Annealing solver ----------------------------------------

class TestAnnealing:
    def test_preserves_palette(
        self, palette: np.ndarray, target_flat: np.ndarray,
    ) -> None:
        mosaic = solve_annealing(
            palette, target_flat,
            color_space="rgb",
            iterations=5_000,
            img_width=W,
            img_height=H,
        )
        palette_set = {tuple(c) for c in palette}
        mosaic_set = {tuple(c) for c in mosaic}
        assert palette_set == mosaic_set

    def test_improves_over_random(
        self, palette: np.ndarray, target_flat: np.ndarray,
    ) -> None:
        rng = np.random.default_rng(0)
        random_mosaic = palette[rng.permutation(N)]
        sa_mosaic = solve_annealing(
            palette, target_flat,
            color_space="rgb",
            iterations=50_000,
            img_width=W,
            img_height=H,
        )
        diff_rand = (target_flat.astype(float) - random_mosaic.astype(float)) ** 2
        random_err = np.mean(np.sqrt(np.sum(diff_rand, axis=1)))
        diff_sa = (target_flat.astype(float) - sa_mosaic.astype(float)) ** 2
        sa_err = np.mean(np.sqrt(np.sum(diff_sa, axis=1)))
        assert sa_err < random_err


# -- Dithering ---------------------------------------------------------

class TestDithering:
    def test_preserves_palette(
        self, palette: np.ndarray, target: np.ndarray,
    ) -> None:
        mosaic_flat = solve_hungarian(palette, target.reshape(-1, 3), "rgb")
        mosaic = mosaic_flat.reshape(H, W, 3)
        dithered = apply_dithering(mosaic, target)
        palette_set = {tuple(c) for c in palette}
        dithered_set = {tuple(c) for c in dithered.reshape(-1, 3)}
        assert dithered_set.issubset(palette_set)

    def test_output_shape(
        self, palette: np.ndarray, target: np.ndarray,
    ) -> None:
        mosaic = palette.reshape(H, W, 3)
        dithered = apply_dithering(mosaic, target)
        assert dithered.shape == (H, W, 3)
        assert dithered.dtype == np.uint8


# -- Image I/O ---------------------------------------------------------

class TestImageIO:
    def test_load_preserves_aspect(self, tmp_image: Path) -> None:
        # tmp_image is 64x48 (w x h)
        arr = load_and_resize(tmp_image, max_side=32)
        h, w = arr.shape[:2]
        assert max(w, h) == 32
        assert w == 32  # landscape: width is longest
        assert h == 24

    def test_save_upscaled(self, tmp_path: Path) -> None:
        arr = np.random.randint(0, 256, (6, 10, 3), dtype=np.uint8)
        out = tmp_path / "test_upscaled.png"
        save_upscaled(arr, out, pixel_upscale=4)
        assert out.exists()
        img = Image.open(out)
        assert img.size == (40, 24)  # 10*4, 6*4
