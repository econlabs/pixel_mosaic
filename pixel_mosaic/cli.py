"""Rich command-line interface powered by Typer."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from pixel_mosaic.config import MosaicConfig
from pixel_mosaic.dithering import apply_dithering
from pixel_mosaic.image_io import load_and_resize, make_comparison_grid, save_upscaled
from pixel_mosaic.palette import (
    extract_palette_from_image,
    generate_palette,
)
from pixel_mosaic.solver_annealing import solve_annealing
from pixel_mosaic.solver_hungarian import solve_hungarian

app = typer.Typer(
    name="pixel-mosaic",
    help="Generate Richter-style pixel mosaics from any image.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
    )


def _collect_images(folder: Path, extensions: frozenset[str]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )


def _quality_metric(target: np.ndarray, mosaic: np.ndarray) -> float:
    t = target.reshape(-1, 3).astype(np.float64)
    m = mosaic.reshape(-1, 3).astype(np.float64)
    return float(np.mean(np.sqrt(np.sum((t - m) ** 2, axis=1))))


# Defaults come from MosaicConfig - single source of truth
_DEFAULTS = MosaicConfig()


# -- batch command -----------------------------------------------------

@app.command()
def batch(
    input_dir: Path = typer.Option(
        _DEFAULTS.input_dir, "--input", "-i", help="Folder with source images",
    ),
    output_dir: Path = typer.Option(
        _DEFAULTS.output_dir, "--output", "-o", help="Results folder",
    ),
    max_side: int = typer.Option(
        _DEFAULTS.max_side, "--max-side", "-m",
        help="Longest side of downscaled image (aspect ratio preserved)",
    ),
    seed: int | None = typer.Option(
        _DEFAULTS.seed, "--seed", "-s", help="Random seed (None = random)",
    ),
    solver: str = typer.Option(
        _DEFAULTS.solver, "--solver", help="'hungarian' or 'annealing'",
    ),
    color_space: str = typer.Option(
        _DEFAULTS.color_space, "--color-space", help="'lab' or 'rgb'",
    ),
    palette_mode: str = typer.Option(
        _DEFAULTS.palette_mode, "--palette", "-p",
        help="Palette: preset name, anchor palette name, or 'anchors'",
    ),
    anchors: str | None = typer.Option(
        None, "--anchors",
        help="Comma-separated hex colours, e.g. '#FF7F11,#262626'",
    ),
    spread: float = typer.Option(
        12.0, "--spread",
        help="Anchor spread in CIELAB units (~5=tight, ~25=wide)",
    ),
    dither: bool = typer.Option(
        _DEFAULTS.dither, "--dither/--no-dither", help="Floyd-Steinberg dithering",
    ),
    upscale: int = typer.Option(
        _DEFAULTS.pixel_upscale, "--upscale", "-u", help="Pixel upscale factor",
    ),
    palette_source: Path | None = typer.Option(
        None, "--palette-from", help="Extract palette from image",
    ),
    sa_iterations: int = typer.Option(
        _DEFAULTS.sa_iterations, "--sa-iter", help="SA iterations",
    ),
    sa_gif: bool = typer.Option(
        True, "--sa-gif/--no-sa-gif", help="Save SA convergence GIF",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Process all images in INPUT_DIR and write results to OUTPUT_DIR."""
    _setup_logging(verbose)
    logger = logging.getLogger("pixel_mosaic")

    cfg = MosaicConfig(
        max_side=max_side,
        seed=seed,
        solver=solver,
        color_space=color_space,
        palette_mode=palette_mode,
        dither=dither,
        pixel_upscale=upscale,
        palette_source=str(palette_source) if palette_source else None,
        sa_iterations=sa_iterations,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    images = _collect_images(input_dir, cfg.SUPPORTED_EXTENSIONS)
    if not images:
        console.print(f"\n[yellow]No images found in {input_dir}/[/yellow]")
        console.print("Place .jpg / .png / ... files there and re-run.\n")
        raise typer.Exit(0)

    console.print(Panel.fit(
        f"[bold]PIXEL MOSAIC GENERATOR[/bold]\n"
        f"Max side: {cfg.max_side}  |  Palette: {cfg.palette_mode}\n"
        f"Solver: {cfg.solver}  |  Colour space: {cfg.color_space}\n"
        f"Dithering: {cfg.dither}  |  Images: {len(images)}",
        border_style="cyan",
    ))

    # Process each image (palette generated per image since dimensions vary)
    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        console.rule(f"[bold cyan][{idx}/{len(images)}] {img_path.name}[/bold cyan]")
        t_total = time.perf_counter()

        # Load & resize (preserving aspect ratio)
        target = load_and_resize(img_path, max_side)
        h, w = target.shape[:2]
        num_pixels = h * w
        target_flat = target.reshape(-1, 3)
        logger.info("Target: %dx%d = %d pixels", w, h, num_pixels)

        if cfg.save_target:
            save_upscaled(
                target, output_dir / f"{stem}_target.{cfg.output_format}", upscale,
            )

        # Generate palette matching this image's pixel count
        if palette_source and palette_source.exists():
            palette = extract_palette_from_image(
                palette_source, num_pixels, seed=seed,
            )
            logger.info("Palette extracted from %s", palette_source)
        else:
            anchor_list = (
                [a.strip() for a in anchors.split(",")]
                if anchors else None
            )
            palette = generate_palette(
                num_pixels, mode=palette_mode, seed=seed,
                anchors=anchor_list, spread=spread,
            )
            logger.info("Palette: %s (%d colours, seed=%s)", palette_mode, num_pixels, seed)

        if cfg.save_palette:
            p_path = output_dir / f"{stem}_palette.{cfg.output_format}"
            save_upscaled(palette.reshape(h, w, 3), p_path, upscale)

        # Solve
        if cfg.solver == "annealing":
            gif_path = output_dir / f"{stem}_annealing.gif" if sa_gif else None
            mosaic_flat = solve_annealing(
                palette, target_flat,
                color_space=cfg.color_space,
                iterations=cfg.sa_iterations,
                gif_path=gif_path,
                img_width=w,
                img_height=h,
                pixel_upscale=upscale,
            )
        else:
            mosaic_flat = solve_hungarian(
                palette, target_flat, color_space=cfg.color_space,
            )

        mosaic = mosaic_flat.reshape(h, w, 3)

        # Dithering
        if cfg.dither:
            logger.info("Applying Floyd-Steinberg dithering ...")
            mosaic = apply_dithering(mosaic, target)

        # Save mosaic
        mosaic_path = output_dir / f"{stem}_mosaic.{cfg.output_format}"
        save_upscaled(mosaic, mosaic_path, upscale)

        # Comparison grid
        if cfg.save_comparison:
            comp_path = output_dir / f"{stem}_comparison.{cfg.output_format}"
            make_comparison_grid(
                img_path, target, mosaic, palette, comp_path, upscale,
            )

        err = _quality_metric(target, mosaic)
        elapsed = time.perf_counter() - t_total

        console.print(
            f"  [green]✓[/green] {mosaic_path.name}  "
            f"[dim]{w}x{h} = {num_pixels} px  error={err:.1f}"
            f"  time={elapsed:.1f}s[/dim]"
        )

    console.print(Panel.fit(
        f"[bold green]ALL DONE[/bold green] - results in [bold]{output_dir}/[/bold]",
        border_style="green",
    ))


# -- single-image command ----------------------------------------------

@app.command()
def single(
    target: Path = typer.Argument(..., help="Path to the target image"),
    output: Path = typer.Option(Path("output/mosaic.png"), "--output", "-o"),
    max_side: int = typer.Option(_DEFAULTS.max_side, "--max-side", "-m"),
    seed: int | None = typer.Option(_DEFAULTS.seed, "--seed", "-s"),
    solver: str = typer.Option(_DEFAULTS.solver, "--solver"),
    color_space: str = typer.Option(_DEFAULTS.color_space, "--color-space"),
    palette_mode: str = typer.Option(_DEFAULTS.palette_mode, "--palette", "-p"),
    anchors: str | None = typer.Option(None, "--anchors"),
    spread: float = typer.Option(12.0, "--spread"),
    dither: bool = typer.Option(_DEFAULTS.dither, "--dither/--no-dither"),
    upscale: int = typer.Option(_DEFAULTS.pixel_upscale, "--upscale", "-u"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process a single image."""
    _setup_logging(verbose)

    output.parent.mkdir(parents=True, exist_ok=True)

    img = load_and_resize(target, max_side)
    h, w = img.shape[:2]
    num_pixels = h * w
    target_flat = img.reshape(-1, 3)

    anchor_list = (
        [a.strip() for a in anchors.split(",")]
        if anchors else None
    )
    palette = generate_palette(
        num_pixels, mode=palette_mode, seed=seed,
        anchors=anchor_list, spread=spread,
    )

    if solver == "annealing":
        mosaic_flat = solve_annealing(
            palette, target_flat,
            color_space=color_space,
            img_width=w,
            img_height=h,
            pixel_upscale=upscale,
        )
    else:
        mosaic_flat = solve_hungarian(palette, target_flat, color_space=color_space)

    mosaic = mosaic_flat.reshape(h, w, 3)

    if dither:
        mosaic = apply_dithering(mosaic, img)

    save_upscaled(mosaic, output, upscale)

    err = _quality_metric(img, mosaic)
    console.print(
        f"[green]✓[/green] Saved to {output}  "
        f"[dim]{w}x{h} = {num_pixels} px  error={err:.1f}[/dim]"
    )


if __name__ == "__main__":
    app()
