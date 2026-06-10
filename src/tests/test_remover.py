import numpy as np
import pytest

from src.core.remover import remove_watermark


def test_remove_watermark_returns_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = remove_watermark(image)

    assert result is not None


def test_remove_watermark_preserves_shape():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = remove_watermark(image)

    assert result.shape == image.shape


def test_remove_watermark_accepts_numpy_array():
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    result = remove_watermark(image)

    assert isinstance(result, np.ndarray)


def test_remove_watermark_invalid_input():
    with pytest.raises(Exception):
        remove_watermark(None)
