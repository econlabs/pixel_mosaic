#!/usr/bin/env python3
"""
main.py — Quick-start entry point.

Drop images into ``images/`` and run:

    python main.py

Or use the full CLI:

    python -m pixel_mosaic.cli batch --help
    python -m pixel_mosaic.cli single my_photo.jpg
"""

from pixel_mosaic.cli import app

if __name__ == "__main__":
    app()
