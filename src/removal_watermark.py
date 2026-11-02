"""Backwards-compatible command wrapper for mask-based inpainting."""
from __future__ import annotations

import argparse

import cv2

from src.core.image_loader import load_image, save_image
from src.core.remover import remove_watermark


def remove_watermark_file(image_path, output_path, mask_path=None, method="telea", radius=3.0):
    if not mask_path:
        raise ValueError("A mask image is required. Create a white-on-black mask for the artifact to repair.")
    image = load_image(image_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    result = remove_watermark(image, mask=mask, method=method, radius=radius)
    return save_image(result, output_path)


def main():
    parser = argparse.ArgumentParser(description="Repair a user-specified image region with mask-based inpainting.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--mask", required=True, help="Grayscale mask; white pixels are repaired")
    parser.add_argument("--method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()
    print(remove_watermark_file(args.input, args.output, args.mask, args.method, args.radius))


if __name__ == "__main__":
    main()
