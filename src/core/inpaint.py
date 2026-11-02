"""Mask-based image inpainting using OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.image_loader import validate_image


def inpaint_image(image, mask, method: str = "telea", radius: float = 3.0):
    """Repair pixels marked white in mask.

    The mask must be supplied explicitly; automatic watermark guessing is
    intentionally avoided because it can erase legitimate image content.
    """
    image = validate_image(image)
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        if mask_array.shape[2] == 4:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGRA2GRAY)
        else:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    if mask_array.ndim != 2 or mask_array.shape != image.shape[:2]:
        raise ValueError("mask must be a 2D array with the same height and width as image")
    if not np.isfinite(mask_array).all():
        raise ValueError("mask must contain finite values")
    binary_mask = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    if not np.any(binary_mask):
        return image.copy()
    if not np.isfinite(radius) or radius <= 0 or radius > 20:
        raise ValueError("radius must be greater than 0 and no more than 20")
    methods = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
    if method not in methods:
        raise ValueError("method must be 'telea' or 'ns'")

    if image.ndim == 3 and image.shape[2] == 4:
        restored = cv2.inpaint(image[:, :, :3], binary_mask, float(radius), methods[method])
        return np.dstack((restored, image[:, :, 3]))
    return cv2.inpaint(image, binary_mask, float(radius), methods[method])
