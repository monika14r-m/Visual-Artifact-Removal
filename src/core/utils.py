# tests/test_utils.py

import os
import numpy as np
import pytest

from src.core.utils import (
    is_image_file,
    list_images,
    resize_image
)


def test_is_image_file_valid():
    assert is_image_file(
        "photo.jpg",
        (".jpg", ".jpeg", ".png")
    )


def test_is_image_file_invalid():
    assert not is_image_file(
        "document.txt",
        (".jpg", ".jpeg", ".png")
    )


def test_list_images_returns_only_images(tmp_path):
    (tmp_path / "image1.jpg").touch()
    (tmp_path / "image2.png").touch()
    (tmp_path / "notes.txt").touch()

    images = list_images(
        str(tmp_path),
        (".jpg", ".jpeg", ".png")
    )

    assert len(images) == 2


def test_list_images_folder_not_found():
    with pytest.raises(FileNotFoundError):
        list_images(
            "nonexistent_folder",
            (".jpg", ".jpeg", ".png")
        )


def test_resize_image_changes_width():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_image(image, 100)

    assert resized.shape[1] == 100


def test_resize_image_preserves_aspect_ratio():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_image(image, 100)

    expected_height = 50

    assert resized.shape[0] == expected_height
