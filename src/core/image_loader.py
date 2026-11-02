"""Safe OpenCV image loading and saving helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    """Validate an OpenCV image array and return it unchanged."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("image must be a non-empty grayscale or color array")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("image must have 1, 3, or 4 channels")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixel values in the 0–255 range")
    return image


def load_image(path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Load an image, preserving its original number of channels."""
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path), flags)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return validate_image(image)


def save_image(image: np.ndarray, path, jpeg_quality: int = 95) -> Path:
    """Save an image, create its parent folder, and raise on write failure."""
    validate_image(image)
    destination = Path(path).expanduser()
    if not destination.suffix:
        raise ValueError("Output path must include an image extension")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    success = cv2.imwrite(str(destination), image, params)
    if not success:
        raise OSError(f"Could not save image: {destination}")
    return destination
