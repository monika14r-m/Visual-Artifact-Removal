"""Image-quality metrics for before/after evaluation."""
from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def _validate_pair(reference, candidate):
    ref = np.asarray(reference)
    test = np.asarray(candidate)
    if ref.shape != test.shape or ref.size == 0:
        raise ValueError("reference and candidate must be non-empty arrays with identical shapes")
    if ref.dtype != np.uint8 or test.dtype != np.uint8:
        raise ValueError("quality metrics expect uint8 images")
    if ref.ndim not in (2, 3):
        raise ValueError("quality metrics expect grayscale or color images")
    return ref, test


def calculate_psnr(reference, candidate) -> float:
    """Calculate peak signal-to-noise ratio in dB (infinity for identical images)."""
    ref, test = _validate_pair(reference, candidate)
    difference = ref.astype(np.float64) - test.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse == 0:
        return math.inf
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def calculate_ssim(reference, candidate) -> float:
    """Calculate structural similarity (SSIM) using scikit-image."""
    ref, test = _validate_pair(reference, candidate)
    channel_axis = -1 if ref.ndim == 3 else None
    value = structural_similarity(ref, test, data_range=255, channel_axis=channel_axis)
    return float(value)


def compare_quality(reference, candidate) -> dict:
    """Return common full-reference image similarity metrics."""
    return {
        "psnr_db": calculate_psnr(reference, candidate),
        "ssim": calculate_ssim(reference, candidate),
    }
