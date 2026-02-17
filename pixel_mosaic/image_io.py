"""Image loading, saving, and comparison-grid generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compute_target_size(
    original_width: int,
    original_height: int,
    max_side: int,
) -> tuple[int, int]:
    """Compute downscaled (w, h) preserving aspect ratio.

    The longest side becomes *max_side*; the other is scaled
    proportionally (rounded to the nearest integer, minimum 1).
    """
    if original_width >= original_height:
        w = max_side
        h = max(1, round(original_height * max_side / original_width))
    else:
        h = max_side
        w = max(1, round(original_width * max_side / original_height))
    return w, h


def load_and_resize(path: str | Path, max_side: int = 32) -> np.ndarray:
    """Load an image and resize preserving aspect ratio.

    The longest side becomes *max_side*.

    Returns:
        (H, W, 3) uint8 array.
    """
    img = Image.open(path).convert("RGB")
    w, h = compute_target_size(img.width, img.height, max_side)
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def save_upscaled(
    array: np.ndarray,
    path: str | Path,
    pixel_upscale: int = 12,
) -> None:
    """Save a small array as a nearest-neighbour-upscaled image."""
    img = Image.fromarray(array.astype(np.uint8))
    h, w = array.shape[:2]
    img = img.resize((w * pixel_upscale, h * pixel_upscale), Image.NEAREST)
    img.save(path)


def make_comparison_grid(
    original_path: str | Path,
    target: np.ndarray,
    mosaic: np.ndarray,
    palette: np.ndarray,
    output_path: str | Path,
    pixel_upscale: int = 12,
) -> None:
    """Create a 4-panel comparison: Original | Target | Palette | Mosaic.

    All panels are upscaled to the same pixel dimensions based on the
    target shape and *pixel_upscale*.
    """
    th, tw = target.shape[:2]
    panel_w = tw * pixel_upscale
    panel_h = th * pixel_upscale
    label_height = 36

    # Prepare panels
    original = (
        Image.open(original_path)
        .convert("RGB")
        .resize((panel_w, panel_h), Image.LANCZOS)
    )
    target_img = Image.fromarray(target).resize((panel_w, panel_h), Image.NEAREST)
    palette_img = Image.fromarray(
        palette.reshape(th, tw, 3)
    ).resize((panel_w, panel_h), Image.NEAREST)
    mosaic_img = Image.fromarray(mosaic).resize((panel_w, panel_h), Image.NEAREST)

    panels = [original, target_img, palette_img, mosaic_img]
    labels = [
        "Original",
        f"Target {tw}x{th}",
        "Palette",
        "Mosaic",
    ]

    gap = 8
    total_w = len(panels) * panel_w + (len(panels) - 1) * gap
    total_h = panel_h + label_height

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18,
        )
    except OSError:
        font = ImageFont.load_default()

    for i, (panel, label) in enumerate(zip(panels, labels, strict=False)):
        x = i * (panel_w + gap)
        canvas.paste(panel, (x, label_height))

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        tx = x + (panel_w - text_w) // 2
        draw.text((tx, 6), label, fill=(220, 220, 220), font=font)

    canvas.save(output_path)
