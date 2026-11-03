"""Streamlit before/after and batch image-restoration interface.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.ai.onnx_restorer import ONNXRestorer
from src.core.remover import restore_image


IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def unique_archive_name(filename: str, used_names: set[str]) -> str:
    """Return a collision-free ZIP entry name for restored images."""
    stem = Path(filename).stem or "image"
    candidate = f"{stem}-restored.png"
    index = 2
    while candidate.casefold() in used_names:
        candidate = f"{stem}-restored-{index}.png"
        index += 1
    used_names.add(candidate.casefold())
    return candidate


def decode_image(data: bytes, grayscale: bool = False):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Images must be 25 MB or smaller.")
    if not data:
        raise ValueError("The uploaded image is empty.")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    if not grayscale and image.dtype != np.uint8:
        raise ValueError("Only 8-bit images are supported by this restoration pipeline.")
    return image


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the restored image.")
    return encoded.tobytes()


def display_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_onnx(image, model_bytes: bytes, task: str, mask=None):
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "uploaded-model.onnx"
        model_path.write_bytes(model_bytes)
        engine = ONNXRestorer(model_path, task=task)
        return engine.upscale(image) if task == "upscale" else engine.inpaint(image, mask)


def process_classical(image, operation: str, strength: float, mask=None):
    return restore_image(image, operation=operation, strength=strength, mask=mask)


def show_comparison(original, restored, file_name="restored.png"):
    before, after = st.columns(2)
    with before:
        st.markdown("**Before**")
        st.image(display_rgb(original), use_container_width=True)
    with after:
        st.markdown("**After**")
        st.image(display_rgb(restored), use_container_width=True)
    st.download_button(
        "Download restored PNG",
        data=encode_png(restored),
        file_name=file_name,
        mime="image/png",
        use_container_width=True,
    )


st.set_page_config(page_title="Visual Artifact Removal", page_icon="🖼️", layout="wide")
st.title("Visual Artifact Removal")
st.caption("Classical image cleanup, optional ONNX deep-learning models, and batch processing.")
st.info(
    "AI models are not bundled. Upload a compatible ONNX export that you trust. "
    "Image tensors use RGB float32 in [0, 1]; inpainting models also need a one-channel mask."
)

single_tab, batch_tab, model_tab = st.tabs(
    ["Single image", "Batch processing", "Model compatibility"]
)

with single_tab:
    uploaded = st.file_uploader("Upload an image", type=IMAGE_TYPES, key="single_image")
    operation = st.selectbox(
        "Restoration method",
        [
            "Denoise (OpenCV)",
            "Deblock (OpenCV)",
            "Enhance (OpenCV)",
            "Mask inpainting (OpenCV)",
            "AI super-resolution (ONNX)",
            "AI inpainting (ONNX)",
        ],
    )
    needs_mask = operation in {"Mask inpainting (OpenCV)", "AI inpainting (ONNX)"}
    mask_upload = (
        st.file_uploader(
            "Upload a black-and-white repair mask (white = repair)",
            type=IMAGE_TYPES,
            key="single_mask",
        )
        if needs_mask else None
    )
    needs_model = operation.startswith("AI ")
    model_upload = (
        st.file_uploader("Upload a trusted ONNX model", type=["onnx"], key="single_model")
        if needs_model else None
    )
    strength = st.slider(
        "Strength", min_value=0, max_value=30, value=7,
        help="Used by denoise, deblock, and enhance.",
    )
    run_single = st.button(
        "Restore image", type="primary", disabled=uploaded is None, use_container_width=True
    )

    if run_single:
        try:
            source = decode_image(uploaded.getvalue())
            mask = decode_image(mask_upload.getvalue(), grayscale=True) if mask_upload else None
            if needs_mask and mask is None:
                raise ValueError("Upload a mask first. White pixels identify the region to repair.")
            if needs_model and model_upload is None:
                raise ValueError("Upload a compatible ONNX model first.")
            if operation == "AI super-resolution (ONNX)":
                result = run_onnx(source, model_upload.getvalue(), "upscale")
            elif operation == "AI inpainting (ONNX)":
                result = run_onnx(source, model_upload.getvalue(), "inpaint", mask)
            else:
                operation_key = {
                    "Denoise (OpenCV)": "denoise",
                    "Deblock (OpenCV)": "deblock",
                    "Enhance (OpenCV)": "enhance",
                    "Mask inpainting (OpenCV)": "inpaint",
                }[operation]
                result = process_classical(source, operation_key, strength, mask)
            st.session_state["single_original"] = source
            st.session_state["single_result"] = result
            st.session_state["single_name"] = Path(uploaded.name).stem + "-restored.png"
            st.success("Restoration completed.")
        except Exception as exc:
            st.error(str(exc))

    if "single_result" in st.session_state:
        show_comparison(
            st.session_state["single_original"],
            st.session_state["single_result"],
            st.session_state.get("single_name", "restored.png"),
        )

with batch_tab:
    batch_files = st.file_uploader(
        "Choose multiple images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="batch_images",
    )
    batch_operation = st.selectbox(
        "Batch operation", ["Denoise", "Deblock", "Enhance"], key="batch_operation"
    )
    batch_strength = st.slider("Batch strength", 0, 30, 7, key="batch_strength")
    run_batch = st.button("Process batch", disabled=not batch_files, use_container_width=True)
    if run_batch:
        output_zip = io.BytesIO()
        rows = []
        used_archive_names = set()
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for uploaded_file in batch_files:
                try:
                    source = decode_image(uploaded_file.getvalue())
                    key = {
                        "Denoise": "denoise",
                        "Deblock": "deblock",
                        "Enhance": "enhance",
                    }[batch_operation]
                    result = process_classical(source, key, batch_strength)
                    archive.writestr(
                        unique_archive_name(uploaded_file.name, used_archive_names), encode_png(result)
                    )
                    rows.append({
                        "file": uploaded_file.name,
                        "status": "completed",
                        "input": f"{source.shape[1]} × {source.shape[0]}",
                        "output": f"{result.shape[1]} × {result.shape[0]}",
                    })
                except Exception as exc:
                    rows.append({
                        "file": uploaded_file.name,
                        "status": f"failed: {exc}",
                        "input": "—",
                        "output": "—",
                    })
        st.session_state["batch_report"] = rows
        st.session_state["batch_zip"] = output_zip.getvalue()

    if "batch_report" in st.session_state:
        completed = sum(row["status"] == "completed" for row in st.session_state["batch_report"])
        st.metric("Successfully restored", f"{completed}/{len(st.session_state['batch_report'])}")
        st.dataframe(st.session_state["batch_report"], use_container_width=True, hide_index=True)
        if st.session_state.get("batch_zip"):
            st.download_button(
                "Download batch ZIP",
                data=st.session_state["batch_zip"],
                file_name="restored-images.zip",
                mime="application/zip",
                use_container_width=True,
            )

with model_tab:
    st.subheader("Supported model contracts")
    st.markdown(
        """
        **Super-resolution / image restoration**
        - One image input and one image output.
        - Rank-4 NCHW or NHWC tensors with three RGB channels.
        - Input values are float32 in the 0–1 range.
        - Output can be 0–1, -1–1, or 0–255.
        - Dynamic height and width are recommended. Fixed-size models only accept their declared input size.

        **LaMa-style mask inpainting**
        - Two inputs: an RGB image and a single-channel mask.
        - The mask input name should include the word mask; otherwise the second input is treated as the mask.
        - White mask pixels indicate the area to reconstruct.
        - Output dimensions must match the source image.

        The repository does not include pretrained weights. Use model files from a source whose
        license permits your use, and verify the model tensor contract before running it.
        The app does not claim that every ONNX export is compatible.
        """
    )
