"""Palette generation: presets, anchors, custom ranges, or extracted from image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

# All available preset palette modes
PALETTE_MODES = [
    "random",
    "pastel",
    "grayscale",
    "warm",
    "cool",
    "earth",
    "neon",
    "sunset",
    "ocean",
    "forest",
    "monochrome_red",
    "monochrome_blue",
    "monochrome_green",
    "vintage",
    "candy",
    "noir",
    "ice",
    "autumn",
    "cyberpunk",
    "terracotta",
    "custom",
]

# Named anchor palettes: name -> (hex_list, default_spread)
ANCHOR_PALETTES: dict[str, tuple[list[str], float]] = {
    "ember": (["#FF7F11", "#ACBFA4", "#E2E8CE", "#262626"], 12.0),
    "ultraviolet": (["#1C0770", "#261CC1", "#3A9AFF", "#F1FF5E"], 12.0),
    "arctic": (["#0B2D72", "#0992C2", "#0AC4E0", "#F6E7BC"], 12.0),
    "cobalt": (["#0D1A63", "#1A2CA3", "#2845D6", "#F68048"], 12.0),
    "solstice": (["#3D45AA", "#DA3D20", "#F8843F", "#FFF19B"], 12.0),
    "overcast": (["#EAEFEF", "#BFC9D1", "#25343F", "#FF9B51"], 12.0),
    "lagoon": (["#09637E", "#088395", "#7AB2B2", "#EBF4F6"], 12.0),
    "harbour": (["#1A3263", "#547792", "#FAB95B", "#E8E2DB"], 12.0),
    "furnace": (["#280905", "#740A03", "#C3110C", "#E6501B"], 12.0),
    "slate": (["#30364F", "#ACBAC4", "#E1D9BC", "#F0F0DB"], 12.0),
    "voltage": (["#362F4F", "#5B23FF", "#008BFF", "#E4FF30"], 12.0),
    "parchment": (["#FAF3E1", "#F5E7C6", "#FA8112", "#222222"], 12.0),
    "depth": (["#0C2C55", "#296374", "#629FAD", "#EDEDCE"], 12.0),
    "phantom": (["#000000", "#9929EA", "#FF5FCF", "#FAEB92"], 12.0),
    "neonoir": (["#00F7FF", "#B0FFFA", "#FF0087", "#FF7DB0"], 12.0),
    "admiral": (["#001F3D", "#ED985F", "#F7B980", "#E6E6E6"], 12.0),
}


def generate_palette(
    num_colors: int,
    mode: str = "random",
    seed: int | None = None,
    custom_ranges: dict[str, tuple[int, int]] | None = None,
    anchors: list[str] | None = None,
    spread: float = 12.0,
) -> np.ndarray:
    """Generate a themed colour palette.

    Args:
        num_colors: How many colours to produce.
        mode: Palette style - one of :data:`PALETTE_MODES`, a key in
            :data:`ANCHOR_PALETTES`, or ``"anchors"`` for custom hex anchors.
        seed: Reproducibility seed (``None`` = non-deterministic).
        custom_ranges: For mode ``"custom"`` only.
        anchors: For mode ``"anchors"`` only. List of hex colour strings
            like ``["#FF7F11", "#262626"]``.
        spread: Spread radius in CIELAB units for anchor interpolation.
            ~5 = tight, ~15 = moderate, ~25 = wide.

    Returns:
        (num_colors, 3) uint8 array.
    """
    rng = np.random.default_rng(seed)

    # Named anchor palette?
    if mode in ANCHOR_PALETTES:
        hex_list, default_spread = ANCHOR_PALETTES[mode]
        return _expand_anchors(num_colors, rng, hex_list, spread or default_spread)

    # Custom hex anchors?
    if mode == "anchors":
        if not anchors:
            msg = "Mode 'anchors' requires anchors=[...] hex list"
            raise ValueError(msg)
        return _expand_anchors(num_colors, rng, anchors, spread)

    if mode == "custom":
        return _custom(num_colors, rng, custom_ranges)

    generators = {
        "random": _random,
        "pastel": _pastel,
        "grayscale": _grayscale,
        "warm": _warm,
        "cool": _cool,
        "earth": _earth,
        "neon": _neon,
        "sunset": _sunset,
        "ocean": _ocean,
        "forest": _forest,
        "monochrome_red": _mono_red,
        "monochrome_blue": _mono_blue,
        "monochrome_green": _mono_green,
        "vintage": _vintage,
        "candy": _candy,
        "noir": _noir,
        "ice": _ice,
        "autumn": _autumn,
        "cyberpunk": _cyberpunk,
        "terracotta": _terracotta,
    }

    gen = generators.get(mode)
    if gen is None:
        available = ", ".join(sorted(generators.keys()))
        msg = f"Unknown palette mode '{mode}'. Available: {available}"
        raise ValueError(msg)

    return gen(num_colors, rng)


# Backward compat
def generate_random_palette(
    num_colors: int, seed: int | None = None,
) -> np.ndarray:
    """Generate *num_colors* uniformly random RGB values."""
    return generate_palette(num_colors, mode="random", seed=seed)


# -- Custom range palette ----------------------------------------------


def _custom(
    n: int,
    rng: np.random.Generator,
    ranges: dict[str, tuple[int, int]] | None,
) -> np.ndarray:
    """User-defined RGB ranges per channel.

    Defaults to full range (0-255) for any unspecified channel.
    """
    if ranges is None:
        ranges = {}
    r_lo, r_hi = ranges.get("r", (0, 255))
    g_lo, g_hi = ranges.get("g", (0, 255))
    b_lo, b_hi = ranges.get("b", (0, 255))

    r = rng.integers(r_lo, r_hi + 1, size=n)
    g = rng.integers(g_lo, g_hi + 1, size=n)
    b = rng.integers(b_lo, b_hi + 1, size=n)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


# -- Preset generators -------------------------------------------------


def _random(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(n, 3), dtype=np.uint8)


def _pastel(n: int, rng: np.random.Generator) -> np.ndarray:
    """Light, desaturated colours."""
    base = rng.integers(140, 256, size=(n, 3))
    mean = base.mean(axis=1, keepdims=True)
    return (base * 0.6 + mean * 0.4).astype(np.uint8)


def _grayscale(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pure grayscale with full range."""
    g = rng.integers(0, 256, size=(n, 1), dtype=np.uint8)
    return np.repeat(g, 3, axis=1)


def _warm(n: int, rng: np.random.Generator) -> np.ndarray:
    """Red, orange, yellow, amber tones."""
    r = rng.integers(150, 256, size=n)
    g = rng.integers(40, 200, size=n)
    b = rng.integers(0, 100, size=n)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _cool(n: int, rng: np.random.Generator) -> np.ndarray:
    """Blue, teal, violet, cyan tones."""
    r = rng.integers(0, 120, size=n)
    g = rng.integers(40, 200, size=n)
    b = rng.integers(140, 256, size=n)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _earth(n: int, rng: np.random.Generator) -> np.ndarray:
    """Brown, beige, olive, terracotta tones."""
    r = rng.integers(80, 210, size=n)
    g = rng.integers(50, 170, size=n)
    b = rng.integers(20, 110, size=n)
    rgb = np.stack([r, g, b], axis=1)
    rgb.sort(axis=1)
    return rgb[:, ::-1].astype(np.uint8)


def _neon(n: int, rng: np.random.Generator) -> np.ndarray:
    """Highly saturated, vivid colours."""
    result = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        channels = [
            rng.integers(200, 256),
            rng.integers(0, 80),
            rng.integers(100, 256),
        ]
        rng.shuffle(channels)
        result[i] = channels
    return result


def _sunset(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deep oranges, pinks, purples, dark blues."""
    t = rng.random(n)
    noise = rng.integers(-20, 20, size=n)
    r = (60 + t * 195).astype(np.uint8)
    g = (20 + t * 100 - np.abs(t - 0.5) * 80).clip(0, 255).astype(np.uint8)
    b = (180 - t * 150 + noise).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _ocean(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deep sea blues, teals, seafoam greens."""
    t = rng.random(n)
    r = (10 + t * 120 + rng.integers(-15, 15, size=n)).clip(0, 255)
    g = (60 + t * 180 + rng.integers(-20, 20, size=n)).clip(0, 255)
    b = (120 + t * 135 + rng.integers(-15, 15, size=n)).clip(0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _forest(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deep greens, mossy tones, bark browns, leaf yellows."""
    choice = rng.integers(0, 3, size=n)
    r = np.where(
        choice == 0, rng.integers(20, 80, size=n),
        np.where(choice == 1, rng.integers(90, 160, size=n), rng.integers(140, 210, size=n)),
    )
    g = np.where(
        choice == 0, rng.integers(80, 180, size=n),
        np.where(choice == 1, rng.integers(50, 100, size=n), rng.integers(150, 220, size=n)),
    )
    b = np.where(
        choice == 0, rng.integers(10, 70, size=n),
        np.where(choice == 1, rng.integers(20, 60, size=n), rng.integers(30, 80, size=n)),
    )
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _mono_red(n: int, rng: np.random.Generator) -> np.ndarray:
    """Monochrome reds: black through bright red."""
    t = rng.random(n)
    r = (t * 255).astype(np.uint8)
    g = (t * rng.integers(0, 50, size=n)).clip(0, 255).astype(np.uint8)
    b = (t * rng.integers(0, 40, size=n)).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _mono_blue(n: int, rng: np.random.Generator) -> np.ndarray:
    """Monochrome blues: black through bright blue."""
    t = rng.random(n)
    r = (t * rng.integers(0, 40, size=n)).clip(0, 255).astype(np.uint8)
    g = (t * rng.integers(0, 60, size=n)).clip(0, 255).astype(np.uint8)
    b = (t * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _mono_green(n: int, rng: np.random.Generator) -> np.ndarray:
    """Monochrome greens: black through bright green."""
    t = rng.random(n)
    r = (t * rng.integers(0, 50, size=n)).clip(0, 255).astype(np.uint8)
    g = (t * 255).astype(np.uint8)
    b = (t * rng.integers(0, 40, size=n)).clip(0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _vintage(n: int, rng: np.random.Generator) -> np.ndarray:
    """Faded, desaturated retro tones with slight sepia."""
    base = rng.integers(60, 200, size=(n, 3))
    # Sepia shift: boost R, keep G, reduce B
    r = (base[:, 0] * 1.1 + 20).clip(0, 255)
    g = (base[:, 1] * 0.95 + 10).clip(0, 255)
    b = (base[:, 2] * 0.75).clip(0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _candy(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bright pinks, purples, magentas, baby blues."""
    choice = rng.integers(0, 3, size=n)
    r = np.where(choice == 0, rng.integers(200, 256, size=n),
         np.where(choice == 1, rng.integers(150, 220, size=n), rng.integers(100, 180, size=n)))
    g = np.where(choice == 0, rng.integers(50, 150, size=n),
         np.where(choice == 1, rng.integers(80, 160, size=n), rng.integers(150, 230, size=n)))
    b = np.where(choice == 0, rng.integers(150, 240, size=n),
         np.where(choice == 1, rng.integers(200, 256, size=n), rng.integers(200, 256, size=n)))
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _noir(n: int, rng: np.random.Generator) -> np.ndarray:
    """High-contrast black & white with slight blue tint."""
    # Bimodal: mostly dark or mostly bright
    t = rng.random(n)
    is_bright = t > 0.5
    base = np.where(is_bright, rng.integers(160, 256, size=n), rng.integers(0, 80, size=n))
    r = base
    g = base
    b = (base + rng.integers(0, 15, size=n)).clip(0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _ice(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pale blues, whites, silver, frozen tones."""
    r = rng.integers(170, 240, size=n)
    g = rng.integers(190, 250, size=n)
    b = rng.integers(220, 256, size=n)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _autumn(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deep reds, burnt oranges, golden yellows, dark browns."""
    choice = rng.integers(0, 4, size=n)
    r = np.where(choice == 0, rng.integers(140, 200, size=n),   # red
         np.where(choice == 1, rng.integers(200, 245, size=n),   # orange
         np.where(choice == 2, rng.integers(200, 240, size=n),   # yellow
                  rng.integers(50, 100, size=n))))                # brown
    g = np.where(choice == 0, rng.integers(20, 60, size=n),
         np.where(choice == 1, rng.integers(100, 160, size=n),
         np.where(choice == 2, rng.integers(170, 210, size=n),
                  rng.integers(30, 70, size=n))))
    b = np.where(choice == 0, rng.integers(10, 40, size=n),
         np.where(choice == 1, rng.integers(10, 50, size=n),
         np.where(choice == 2, rng.integers(20, 60, size=n),
                  rng.integers(10, 40, size=n))))
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _cyberpunk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Electric pinks, neon cyans, deep purples, black."""
    choice = rng.integers(0, 4, size=n)
    r = np.where(choice == 0, rng.integers(200, 256, size=n),   # pink
         np.where(choice == 1, rng.integers(0, 40, size=n),      # cyan
         np.where(choice == 2, rng.integers(80, 140, size=n),    # purple
                  rng.integers(0, 30, size=n))))                  # black
    g = np.where(choice == 0, rng.integers(0, 60, size=n),
         np.where(choice == 1, rng.integers(200, 256, size=n),
         np.where(choice == 2, rng.integers(0, 50, size=n),
                  rng.integers(0, 20, size=n))))
    b = np.where(choice == 0, rng.integers(100, 180, size=n),
         np.where(choice == 1, rng.integers(200, 256, size=n),
         np.where(choice == 2, rng.integers(160, 230, size=n),
                  rng.integers(0, 30, size=n))))
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _terracotta(n: int, rng: np.random.Generator) -> np.ndarray:
    """Warm clay, dusty rose, sandstone, muted terracotta."""
    r = rng.integers(140, 220, size=n)
    g = rng.integers(70, 140, size=n)
    b = rng.integers(50, 110, size=n)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


# -- Anchor-based palette expansion ------------------------------------


def _hex_to_rgb(hex_str: str) -> np.ndarray:
    """Parse '#RRGGBB' to (3,) uint8 array."""
    h = hex_str.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _expand_anchors(
    n: int,
    rng: np.random.Generator,
    hex_colors: list[str],
    spread: float,
) -> np.ndarray:
    """Expand a small set of anchor colours to *n* colours.

    Interpolation happens in CIELAB for perceptually uniform results.
    Each generated colour is a weighted blend of two random anchors,
    plus Gaussian noise controlled by *spread*.

    Args:
        n: Number of colours to produce.
        rng: NumPy random generator.
        hex_colors: Anchor colours as hex strings.
        spread: Gaussian noise sigma in CIELAB units.

    Returns:
        (n, 3) uint8 RGB array.
    """
    # Parse anchors to LAB
    rgb_anchors = np.array([_hex_to_rgb(h) for h in hex_colors])
    lab_anchors = rgb2lab(
        rgb_anchors.astype(np.float64).reshape(1, -1, 3) / 255.0,
    ).reshape(-1, 3)

    k = len(lab_anchors)

    # For each output colour: pick two random anchors and a blend weight
    idx_a = rng.integers(0, k, size=n)
    idx_b = rng.integers(0, k, size=n)
    t = rng.random(n).reshape(-1, 1)

    # Interpolate between the two anchors in LAB
    lab_colors = lab_anchors[idx_a] * (1 - t) + lab_anchors[idx_b] * t

    # Add Gaussian spread
    noise = rng.normal(0, spread, size=(n, 3))
    # Less noise on L channel to preserve overall brightness structure
    noise[:, 0] *= 0.6
    lab_colors = lab_colors + noise

    # Clamp L to [0, 100], a/b to [-128, 127]
    lab_colors[:, 0] = np.clip(lab_colors[:, 0], 0, 100)
    lab_colors[:, 1] = np.clip(lab_colors[:, 1], -128, 127)
    lab_colors[:, 2] = np.clip(lab_colors[:, 2], -128, 127)

    # Convert back to RGB
    rgb_float = lab2rgb(lab_colors.reshape(1, -1, 3)).reshape(-1, 3)
    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)

    return rgb_uint8


# -- Extract from image ------------------------------------------------


def extract_palette_from_image(
    path: str | Path,
    num_colors: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample *num_colors* pixel colours from an existing image.

    Args:
        path: Path to the source image.
        num_colors: Desired palette size.
        seed: Reproducibility seed.

    Returns:
        (num_colors, 3) uint8 array.
    """
    rng = np.random.default_rng(seed)
    img = Image.open(path).convert("RGB")

    w, h = img.size
    scale = max(1.0, (num_colors * 4 / (w * h)) ** 0.5)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    pixels = np.array(img).reshape(-1, 3)

    if len(pixels) >= num_colors:
        idx = rng.choice(len(pixels), size=num_colors, replace=False)
    else:
        idx = rng.choice(len(pixels), size=num_colors, replace=True)

    return pixels[idx].astype(np.uint8)
