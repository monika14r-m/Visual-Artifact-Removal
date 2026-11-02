"""Classical image-restoration operations."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.image_loader import validate_image
from src.core.inpaint import inpaint_image


def _split_alpha(image):
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, None


def denoise_image(image, strength: float = 7.0, preserve_edges: bool = True):
    """Reduce random noise with OpenCV non-local means denoising."""
    image = validate_image(image)
    if not np.isfinite(strength) or not 0 <= strength <= 30:
        raise ValueError("strength must be between 0 and 30")
    if strength == 0:
        return image.copy()

    color, alpha = _split_alpha(image)
    if color.ndim == 2 or (color.ndim == 3 and color.shape[2] == 1):
        gray = color if color.ndim == 2 else color[:, :, 0]
        result = cv2.fastNlMeansDenoising(gray, None, h=float(strength), templateWindowSize=7, searchWindowSize=21)
        if color.ndim == 3:
            result = result[:, :, None]
    else:
        h = float(strength)
        result = cv2.fastNlMeansDenoisingColored(
            color, None, h, h * (0.8 if preserve_edges else 1.0),
            templateWindowSize=7, searchWindowSize=21,
        )
    if alpha is not None:
        result = np.dstack((result, alpha))
    return result


def reduce_compression_artifacts(image, strength: float = 5.0):
    """Soften JPEG-like ringing and block edges with edge-preserving filtering.

    This is a lightweight approximation, not a codec-aware deblocking model;
    excessive strength can soften small textures and text.
    """
    image = validate_image(image)
    if not np.isfinite(strength) or not 0 <= strength <= 10:
        raise ValueError("strength must be between 0 and 10")
    if strength == 0:
        return image.copy()

    color, alpha = _split_alpha(image)
    diameter = 5 if strength <= 5 else 7
    sigma_color = 15.0 + float(strength) * 5.0
    sigma_space = 5.0 + float(strength) * 0.8
    if color.ndim == 2 or (color.ndim == 3 and color.shape[2] == 1):
        gray = color if color.ndim == 2 else color[:, :, 0]
        result = cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)
        if color.ndim == 3:
            result = result[:, :, None]
    else:
        result = cv2.bilateralFilter(color, diameter, sigma_color, sigma_space)
    if alpha is not None:
        result = np.dstack((result, alpha))
    return result


def enhance_image(image, amount: float = 0.7, radius: float = 1.2):
    """Apply a restrained unsharp mask to improve perceived edge clarity."""
    image = validate_image(image)
    if not np.isfinite(amount) or not 0 <= amount <= 2:
        raise ValueError("amount must be between 0 and 2")
    if not np.isfinite(radius) or not 0.1 <= radius <= 10:
        raise ValueError("radius must be between 0.1 and 10")
    if amount == 0:
        return image.copy()

    color, alpha = _split_alpha(image)
    blurred = cv2.GaussianBlur(color, (0, 0), sigmaX=float(radius), sigmaY=float(radius))
    sharpened = cv2.addWeighted(color, 1.0 + float(amount), blurred, -float(amount), 0)
    if alpha is not None:
        sharpened = np.dstack((sharpened, alpha))
    return sharpened


def remove_watermark(image, mask=None, method: str = "telea", radius: float = 3.0):
    """Inpaint a user-supplied mask; automatic bright-region guessing is disabled."""
    if mask is None:
        raise ValueError("Provide an explicit mask; automatic watermark detection is not reliable.")
    return inpaint_image(image, mask, method=method, radius=radius)


def restore_image(image, operation: str = "denoise", mask=None, strength: float = 7.0, method: str = "telea", radius: float = 3.0):
    """Run one selected restoration operation."""
    operations = {
        "denoise": lambda: denoise_image(image, strength=strength),
        "deblock": lambda: reduce_compression_artifacts(image, strength=min(float(strength), 10.0)),
        "enhance": lambda: enhance_image(image, amount=min(float(strength) / 10.0, 2.0)),
        "inpaint": lambda: inpaint_image(image, mask, method=method, radius=radius),
    }
    if operation not in operations:
        raise ValueError(f"Unsupported operation '{operation}'. Choose: {', '.join(operations)}")
    return operations[operation]()
