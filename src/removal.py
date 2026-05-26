import cv2
import numpy as np

def remove_watermark(image_path, output_path, mask=None):
    """
    Removes watermark from an image using inpainting.
    :param image_path: Path to input image
    :param output_path: Path to save output image
    :param mask: Optional binary mask (same size as image) where watermark is marked
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    if mask is None:
        # Simple heuristic: assume watermark is bright text in bottom-right corner
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        mask[h-80:h-20, w-200:w-20] = 255

    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, result)
    return output_path
