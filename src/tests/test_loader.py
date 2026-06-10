
import pytest
import numpy as np
import cv2
import os

from src.core.image_loader import load_image


def test_load_valid_image(tmp_path):
    """
    Test loading a valid image.
    """
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(image_path), image)

    loaded = load_image(str(image_path))

    assert loaded is not None
    assert loaded.shape == image.shape


def test_load_nonexistent_image():
    """
    Test loading an image that does not exist.
    """
    with pytest.raises(Exception):
        load_image("does_not_exist.jpg")


def test_load_invalid_file(tmp_path):
    """
    Test loading a non-image file.
    """
    file_path = tmp_path / "dummy.txt"

    with open(file_path, "w") as f:
        f.write("not an image")

    with pytest.raises(Exception):
        load_image(str(file_path))
