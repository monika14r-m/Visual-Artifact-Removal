"""
removal_watermark.py
--------------------
Provides functions to remove watermarks from images using OpenCV inpainting.
Includes a CLI entry point for direct usage.
"""

import cv2
import numpy as np
import argparse
import os


def remove_watermark(image_path, output_path, mask=None):
    """
    Removes watermark from an image using inpainting.

    Parameters
    ----------
    image_path : str
        Path to input image.
    output_path : str
        Path to save output image.
    mask : np.ndarray, optional
        Binary mask (same size as image) where watermark is marked.
        If None, a default heuristic mask is applied.

    Returns
    -------
    str
        Path to the saved output image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    if mask is not None and mask.shape != image.shape[:2]:
        raise ValueError("Mask must match image dimensions")

    if mask is None:
        # Simple heuristic: assume watermark is in bottom-right corner
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        mask[h-80:h-20, w-200:w-20] = 255

    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, result)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Remove watermark from image")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save cleaned image")
    parser.add_argument("--mask", help="Optional path to mask image", default=None)
    args = parser.parse_args()

    mask = None
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {args.mask}")

    out_path = remove_watermark(args.input, args.output, mask)
    print(f"Watermark removed. Saved output to {out_path}")


if __name__ == "__main__":
    main()
