import os
import pytest
from src.remove_watermark import remove_watermark

def test_output_file_created(tmp_path):
    input_img = "examples/sample_with_watermark.jpg"
    output_img = tmp_path / "clean.jpg"
    result = remove_watermark(input_img, str(output_img))
    assert os.path.exists(result)

def test_output_dimensions(tmp_path):
    input_img = "examples/sample_with_watermark.jpg"
    output_img = tmp_path / "clean.jpg"
    result = remove_watermark(input_img, str(output_img))
    import cv2
    inp = cv2.imread(input_img)
    out = cv2.imread(result)
    assert inp.shape == out.shape

def test_invalid_input_raises():
    with pytest.raises(ValueError):
        remove_watermark("nonexistent.jpg", "out.jpg")
