import numpy as np
import pytest
import cv2

from src.core.image_loader import load_image, save_image
from src.core.inpaint import inpaint_image
from src.core.remover import (
    denoise_image,
    enhance_image,
    reduce_compression_artifacts,
    remove_watermark,
    restore_image,
)
from src.core.utils import is_image_file, list_images, resize_image
from src.metrics import calculate_psnr, calculate_ssim, compare_quality


def sample_image(height=48, width=64):
    y, x = np.mgrid[:height, :width]
    base = ((x * 3 + y * 2) % 180 + 40).astype(np.uint8)
    return np.dstack((base, np.roll(base, 4, axis=1), np.roll(base, 7, axis=0)))


def test_load_and_save_round_trip(tmp_path):
    image = sample_image()
    output = save_image(image, tmp_path / "nested" / "sample.png")
    loaded = load_image(output)
    np.testing.assert_array_equal(loaded, image)


def test_loader_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_loader_rejects_invalid_image_data(tmp_path):
    path = tmp_path / "not-an-image.png"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="Could not decode"):
        load_image(path)


def test_denoise_preserves_shape_and_dtype():
    image = sample_image()
    noisy = np.clip(image.astype(np.int16) + np.random.default_rng(2).normal(0, 12, image.shape), 0, 255).astype(np.uint8)
    restored = denoise_image(noisy, strength=8)
    assert restored.shape == noisy.shape
    assert restored.dtype == np.uint8


def test_denoise_preserves_alpha_channel():
    color = sample_image()
    alpha = np.full(color.shape[:2], 173, dtype=np.uint8)
    image = np.dstack((color, alpha))
    restored = denoise_image(image, strength=4)
    np.testing.assert_array_equal(restored[:, :, 3], alpha)


def test_denoise_rejects_invalid_strength():
    with pytest.raises(ValueError):
        denoise_image(sample_image(), strength=40)


def test_deblock_and_enhance_keep_dimensions():
    image = sample_image()
    deblocked = reduce_compression_artifacts(image, strength=4)
    enhanced = enhance_image(image, amount=0.5)
    assert deblocked.shape == image.shape
    assert enhanced.shape == image.shape
    assert deblocked.dtype == enhanced.dtype == np.uint8


def test_zero_strength_is_no_op():
    image = sample_image()
    np.testing.assert_array_equal(denoise_image(image, strength=0), image)
    np.testing.assert_array_equal(reduce_compression_artifacts(image, strength=0), image)
    np.testing.assert_array_equal(enhance_image(image, amount=0), image)


def test_inpaint_uses_explicit_mask_and_preserves_shape():
    image = sample_image()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[18:25, 20:30] = 255
    restored = inpaint_image(image, mask, method="telea", radius=3)
    assert restored.shape == image.shape
    assert restored.dtype == image.dtype


def test_inpaint_rejects_wrong_mask_dimensions():
    with pytest.raises(ValueError, match="same height and width"):
        inpaint_image(sample_image(), np.zeros((10, 10), dtype=np.uint8))


def test_watermark_removal_requires_user_mask():
    with pytest.raises(ValueError, match="explicit mask"):
        remove_watermark(sample_image())


def test_restore_image_rejects_unknown_operation():
    with pytest.raises(ValueError, match="Unsupported operation"):
        restore_image(sample_image(), operation="magic")


def test_image_helpers_and_aspect_ratio_resize(tmp_path):
    assert is_image_file("photo.JPEG")
    assert not is_image_file("notes.txt")
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "b.txt").write_text("skip")
    assert len(list_images(tmp_path)) == 1
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")
    resized = resize_image(sample_image(40, 80), 40)
    assert resized.shape[:2] == (20, 40)


def test_quality_metrics_identical_image():
    image = sample_image(40, 40)
    assert calculate_psnr(image, image) == float("inf")
    assert calculate_ssim(image, image) == pytest.approx(1.0)
    assert compare_quality(image, image)["ssim"] == pytest.approx(1.0)


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        calculate_psnr(sample_image(), sample_image(20, 20))
