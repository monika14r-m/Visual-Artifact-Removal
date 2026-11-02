"""Small filesystem and image utility helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.core.image_loader import validate_image

DEFAULT_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(filename, formats=DEFAULT_FORMATS) -> bool:
    return Path(filename).suffix.lower() in {ext.lower() for ext in formats}


def list_images(directory, formats=DEFAULT_FORMATS, recursive: bool = False):
    folder = Path(directory).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(str(path) for path in iterator if path.is_file() and is_image_file(path.name, formats))


def resize_image(image, width: int):
    image = validate_image(image)
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)
