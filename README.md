# 🎨 Pixel Mosaic Generator

[![CI](https://github.com/YOURNAME/pixel-mosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/YOURNAME/pixel-mosaic/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Generate **Gerhard Richter–style pixel mosaics**: take a fixed set of 4 096 random colours, then rearrange them in a 64×64 grid to best approximate any target image. No colours are added or modified — only their positions change.

<p align="center">
  <img src="docs/comparison_example.png" alt="Example comparison grid" width="100%">
</p>

## ✨ Features

- **Two solvers** — optimal [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) (~20 s) and tuneable [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) with animated GIF export
- **Perceptual colour matching** in CIELAB space (or RGB)
- **Floyd–Steinberg dithering** post-processing for smoother tonal gradients
- **Palette extraction** from any source image (not just random)
- **Batch processing** — drop multiple images in `images/`, run once
- **Comparison grid** — auto-generated side-by-side panels (Original → Target → Palette → Mosaic)
- **Interactive Streamlit demo** for the browser
- **Modern Python** — `pyproject.toml`, type hints, `typer` CLI, `rich` logging
- **CI/CD** — GitHub Actions, ruff, mypy, pytest with coverage
- **Docker support** for reproducible builds

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/YOURNAME/pixel-mosaic.git
cd pixel-mosaic
pip install -e ".[all]"   # includes dev + streamlit deps
```

### Basic usage

```bash
# Batch mode: process everything in images/
cp your_photo.jpg images/
python main.py batch

# Single image
python main.py single photo.jpg -o output/result.png
```

### Streamlit demo

```bash
streamlit run streamlit_app.py
```

## 🛠 CLI Reference

```
pixel-mosaic batch [OPTIONS]      Process all images in a folder
pixel-mosaic single TARGET [OPT]  Process a single image
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--solver` | `hungarian` | `hungarian` (optimal) or `annealing` (heuristic) |
| `--grid` / `-g` | `64` | Grid side length (64 → 4 096 pixels) |
| `--seed` / `-s` | `42` | Random seed for palette |
| `--color-space` | `lab` | `lab` (perceptual) or `rgb` |
| `--dither / --no-dither` | off | Floyd–Steinberg error diffusion |
| `--upscale` / `-u` | `12` | Output pixel size multiplier |
| `--palette-from` | — | Extract palette from an image file |
| `--sa-iter` | `2000000` | Simulated Annealing iterations |
| `--sa-gif / --no-sa-gif` | on | Save SA convergence animation |
| `--verbose` / `-v` | off | Debug logging |

### Examples

```bash
# Annealing solver with animation
python main.py batch --solver annealing --sa-iter 3000000

# Extract palette from a Richter painting
python main.py batch --palette-from richter_4096.jpg

# High-res 128×128 grid (16 384 pixels — slower!)
python main.py single portrait.jpg --grid 128 --upscale 8

# RGB colour space + dithering
python main.py batch --color-space rgb --dither
```

## 🧪 Development

```bash
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest --cov=pixel_mosaic

# Lint
ruff check pixel_mosaic/ tests/
ruff format pixel_mosaic/ tests/

# Type check
mypy pixel_mosaic/
```

## 🐳 Docker

```bash
docker build -t pixel-mosaic .
docker run -v $(pwd)/images:/app/images -v $(pwd)/output:/app/output pixel-mosaic batch
```

## 📐 How It Works

### The Problem

Given *N* = 4 096 colours (randomly generated or sampled) and a target image resized to 64×64 pixels, find the **bijective mapping** from colours → grid positions that minimises the total perceptual colour distance.

### Solvers

**Hungarian Algorithm** — models the problem as a [linear assignment problem](https://en.wikipedia.org/wiki/Assignment_problem). Computes an *N × N* cost matrix of pairwise CIELAB distances, then finds the globally optimal assignment in *O(N³)* time via `scipy.optimize.linear_sum_assignment`.

**Simulated Annealing** — starts from a random permutation and iteratively swaps two pixel positions. Swaps that reduce error are always accepted; worse swaps are accepted with decreasing probability as the "temperature" cools. Produces a GIF showing the image emerge from noise.

### Dithering

An optional Floyd–Steinberg error-diffusion pass redistributes quantisation error to neighbouring pixels via local swaps, producing smoother gradients without introducing any new colours.

## 📁 Project Structure

```
pixel-mosaic/
├── main.py                     # Entry point
├── streamlit_app.py            # Interactive demo
├── pyproject.toml              # Packaging & tool config
├── Dockerfile
├── LICENSE
├── pixel_mosaic/
│   ├── __init__.py             # Public API
│   ├── config.py               # Frozen dataclass config
│   ├── cli.py                  # Typer CLI with Rich output
│   ├── palette.py              # Random + image-based palette gen
│   ├── color_utils.py          # LAB conversion, cost matrix
│   ├── solver_hungarian.py     # Optimal solver (scipy)
│   ├── solver_annealing.py     # SA solver + GIF export
│   ├── dithering.py            # Floyd–Steinberg post-processing
│   └── image_io.py             # Load, save, comparison grid
├── tests/
│   └── test_mosaic.py          # Comprehensive test suite
├── images/                     # ← Drop input images here
├── output/                     # ← Results appear here
└── .github/workflows/ci.yml   # GitHub Actions CI
```

## 📄 License

MIT — see [LICENSE](LICENSE).
