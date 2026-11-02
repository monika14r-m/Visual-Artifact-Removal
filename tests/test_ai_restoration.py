import cv2
import numpy as np
import pytest

from src.ai.onnx_restorer import ONNXRestorer


class FakeNode:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeUpscaleSession:
    def get_inputs(self):
        return [FakeNode("image", [1, 3, None, None])]

    def get_outputs(self):
        return [FakeNode("restored", [1, 3, None, None])]

    def run(self, output_names, feed):
        rgb = np.transpose(feed["image"][0], (1, 2, 0))
        enlarged = cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 2))
        return [np.transpose(enlarged, (2, 0, 1))[None, ...]]


class FakeInpaintSession:
    def get_inputs(self):
        return [
            FakeNode("image", [1, 3, None, None]),
            FakeNode("mask", [1, 1, None, None]),
        ]

    def get_outputs(self):
        return [FakeNode("restored", [1, 3, None, None])]

    def run(self, output_names, feed):
        assert feed["mask"].shape[1] == 1
        return [feed["image"]]


def sample_image():
    y, x = np.mgrid[:24, :32]
    return np.dstack((x * 5, y * 7, (x + y) * 3)).clip(0, 255).astype(np.uint8)


def test_onnx_upscale_adapter_handles_nchw_output():
    image = sample_image()
    engine = ONNXRestorer(task="upscale", session=FakeUpscaleSession())
    result = engine.upscale(image)
    assert result.shape == (48, 64, 3)
    assert result.dtype == np.uint8


def test_onnx_inpaint_adapter_maps_image_and_mask_inputs():
    image = sample_image()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[4:8, 5:9] = 255
    engine = ONNXRestorer(task="inpaint", session=FakeInpaintSession())
    result = engine.inpaint(image, mask)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_onnx_inpaint_rejects_mismatched_mask():
    image = sample_image()
    engine = ONNXRestorer(task="inpaint", session=FakeInpaintSession())
    with pytest.raises(ValueError, match="match the input image"):
        engine.inpaint(image, np.zeros((4, 4), dtype=np.uint8))


def test_onnx_adapter_rejects_wrong_number_of_model_inputs():
    class InvalidSession:
        def get_inputs(self):
            return [FakeNode("a", [1, 3, None, None]), FakeNode("b", [1, 3, None, None])]

        def get_outputs(self):
            return [FakeNode("out", [1, 3, None, None])]

    with pytest.raises(ValueError, match="exactly one image input"):
        ONNXRestorer(task="upscale", session=InvalidSession())
