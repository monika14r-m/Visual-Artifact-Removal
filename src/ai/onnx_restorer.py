"""Generic ONNX image-restoration adapter.

Supports an explicit, portable tensor contract: RGB float32 image tensors in
[0, 1], with NCHW or NHWC layout. Inpainting models take a second single-
channel mask tensor in [0, 1]. Model weights are supplied by the user.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from src.core.image_loader import validate_image


class ONNXRestorer:
    """Run a user-supplied ONNX image-upscaling or mask-inpainting model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        task: str = "upscale",
        providers: Iterable[str] | None = None,
        output_range: str = "auto",
        session: Any | None = None,
    ) -> None:
        if task not in {"upscale", "inpaint"}:
            raise ValueError("task must be 'upscale' or 'inpaint'")
        if output_range not in {"auto", "zero_one", "minus_one_one", "zero_255"}:
            raise ValueError("Unsupported output_range")
        if session is None:
            if model_path is None:
                raise ValueError("model_path is required when no session is supplied")
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "AI restoration requires ONNX Runtime. Install it with: "
                    "pip install -r requirements-ai.txt"
                ) from exc
            session = ort.InferenceSession(
                str(path),
                providers=list(providers) if providers else ["CPUExecutionProvider"],
            )

        self.session = session
        self.task = task
        self.output_range = output_range
        self.inputs = list(session.get_inputs())
        self.outputs = list(session.get_outputs())
        expected_inputs = 1 if task == "upscale" else 2
        if len(self.inputs) != expected_inputs:
            requirement = "one image input" if task == "upscale" else "image and mask inputs"
            raise ValueError(f"This {task} model must expose exactly {requirement}")
        if not self.outputs:
            raise ValueError("The ONNX model has no output tensors")

        self.image_input = self._find_image_input()
        self.mask_input = self._find_mask_input() if task == "inpaint" else None
        self._validate_input_shape(self.image_input, channels=3)
        if self.mask_input is not None:
            self._validate_input_shape(self.mask_input, channels=1)

    def _find_image_input(self):
        if self.task == "upscale":
            return self.inputs[0]
        mask_candidates = [node for node in self.inputs if "mask" in node.name.lower()]
        if len(mask_candidates) == 1:
            return next(node for node in self.inputs if node is not mask_candidates[0])
        return self.inputs[0]

    def _find_mask_input(self):
        if self.task != "inpaint":
            return None
        candidates = [node for node in self.inputs if "mask" in node.name.lower()]
        if len(candidates) == 1:
            return candidates[0]
        return next(node for node in self.inputs if node is not self.image_input)

    @staticmethod
    def _shape_layout(node, channels: int) -> str:
        shape = getattr(node, "shape", None)
        if shape is None or len(shape) != 4:
            raise ValueError(f"Input '{node.name}' must be a rank-4 NCHW or NHWC tensor")
        if shape[1] == channels:
            return "NCHW"
        if shape[-1] == channels:
            return "NHWC"
        raise ValueError(
            f"Input '{node.name}' must have {channels} channels in NCHW or NHWC layout"
        )

    @classmethod
    def _validate_input_shape(cls, node, channels: int) -> None:
        cls._shape_layout(node, channels)
        shape = node.shape
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError(f"Input '{node.name}' must use batch size 1")

    @staticmethod
    def _check_fixed_dimensions(node, height: int, width: int, layout: str) -> None:
        shape = node.shape
        expected_height, expected_width = (
            (shape[2], shape[3]) if layout == "NCHW" else (shape[1], shape[2])
        )
        for label, expected, actual in (
            ("height", expected_height, height),
            ("width", expected_width, width),
        ):
            if isinstance(expected, int) and expected > 0 and expected != actual:
                raise ValueError(
                    f"Model input '{node.name}' requires {label} {expected}; received {actual}. "
                    "Use a model export with dynamic spatial dimensions or matching input size."
                )

    @classmethod
    def _make_tensor(cls, array: np.ndarray, node, channels: int) -> np.ndarray:
        layout = cls._shape_layout(node, channels)
        height, width = array.shape[:2]
        cls._check_fixed_dimensions(node, height, width, layout)
        tensor = array[None, ...] if layout == "NHWC" else np.transpose(array, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    @staticmethod
    def _as_three_channel_bgr(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        image = validate_image(image)
        alpha = None
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            bgr = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr, alpha = image[:, :, :3], image[:, :, 3]
        else:
            bgr = image
        return bgr, alpha

    def _output_to_bgr(self, output: np.ndarray) -> np.ndarray:
        data = np.asarray(output)
        if data.ndim == 4:
            if data.shape[0] != 1:
                raise ValueError("Model output batch size must be 1")
            data = data[0]
        if data.ndim != 3:
            raise ValueError("Model output must be a 3D image tensor or a batch of one")
        if data.shape[0] in (1, 3):
            data = np.transpose(data, (1, 2, 0))
        elif data.shape[-1] not in (1, 3):
            raise ValueError("Model output must contain one or three image channels")
        if data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=2)
        data = data.astype(np.float32, copy=False)
        if not np.isfinite(data).all():
            raise ValueError("Model output contains NaN or infinite values")

        if self.output_range == "minus_one_one":
            data = (data + 1.0) / 2.0
        elif self.output_range == "zero_255":
            data = data / 255.0
        elif self.output_range == "zero_one":
            pass
        else:
            low, high = float(data.min()), float(data.max())
            if low < -0.01 and low >= -1.25 and high <= 1.25:
                data = (data + 1.0) / 2.0
            elif low >= -0.01 and high <= 1.5:
                pass
            elif low >= 0 and high <= 255.5:
                data = data / 255.0
            else:
                raise ValueError(
                    "Could not infer model output range. Set output_range explicitly "
                    "to zero_one, minus_one_one, or zero_255."
                )

        rgb_u8 = np.rint(np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    def _execute(self, image_tensor: np.ndarray, mask_tensor: np.ndarray | None = None) -> np.ndarray:
        feed = {self.image_input.name: image_tensor}
        if self.mask_input is not None:
            if mask_tensor is None:
                raise ValueError("An inpainting mask tensor is required")
            feed[self.mask_input.name] = mask_tensor
        output = self.session.run([self.outputs[0].name], feed)[0]
        return self._output_to_bgr(output)

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Run a one-input RGB restoration/super-resolution ONNX model."""
        bgr, alpha = self._as_three_channel_bgr(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self._make_tensor(rgb, self.image_input, channels=3)
        restored = self._execute(tensor)
        if restored.shape[0] < bgr.shape[0] or restored.shape[1] < bgr.shape[1]:
            raise ValueError("Upscale model output is smaller than its input")
        if alpha is not None:
            resized_alpha = cv2.resize(
                alpha, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            restored = np.dstack((restored, resized_alpha))
        return restored

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run a two-input RGB image-plus-mask ONNX inpainting model."""
        if mask is None:
            raise ValueError("An explicit mask is required for AI inpainting")
        bgr, alpha = self._as_three_channel_bgr(image)
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            if mask_array.shape[2] == 4:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGRA2GRAY)
            else:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
        if mask_array.ndim != 2 or mask_array.shape != bgr.shape[:2]:
            raise ValueError("mask must match the input image height and width")
        if not np.isfinite(mask_array).all():
            raise ValueError("mask must contain finite values")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_float = (mask_array > 0).astype(np.float32)[:, :, None]
        image_tensor = self._make_tensor(rgb, self.image_input, channels=3)
        mask_tensor = self._make_tensor(mask_float, self.mask_input, channels=1)
        restored = self._execute(image_tensor, mask_tensor)
        if restored.shape[:2] != bgr.shape[:2]:
            raise ValueError("AI inpainting model output must match the input image dimensions")
        if alpha is not None:
            restored = np.dstack((restored, alpha))
        return restored
