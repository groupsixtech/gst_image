"""Image loading helpers optimized for large stitched micrographs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_image(path: str | Path, *, color: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def load_preview(
    path: str | Path, max_size: tuple[int, int] = (2400, 1400), *, color: bool = True
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load a reduced preview and return it with full (width, height)."""
    probe = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8 if color else cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if probe is None:
        raise ValueError(f"Could not read image: {path}")
    # Reduced decoding ratios differ by format, so obtain exact metadata via Pillow lazily.
    from PIL import Image

    # Large stitched metallographs are expected inputs, not decompression-bomb payloads.
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(path) as source:
        full_size = source.size
    max_width, max_height = max_size
    scale = min(1.0, max_width / probe.shape[1], max_height / probe.shape[0])
    if scale < 1:
        probe = cv2.resize(
            probe,
            (max(1, round(probe.shape[1] * scale)), max(1, round(probe.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return probe, full_size


def iter_images(directory: str | Path) -> list[Path]:
    return sorted(
        path for path in Path(directory).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
