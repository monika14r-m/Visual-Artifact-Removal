import cv2
import numpy as np

def remove_watermark(image_path, output_path, mask=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    if mask is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        mask[h-80:h-20, w-200:w-20] = 255

    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, result)
    return output_path
