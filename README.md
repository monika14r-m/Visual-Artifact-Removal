# Visual Artifact Removal

A Python image-restoration toolkit with two processing paths: lightweight OpenCV filters and an optional ONNX deep-learning runtime. It includes a command-line interface, a Streamlit before/after workspace, mask-based repair, and ZIP batch export.

The project is designed for reversible, transparent processing. It does not promise to reconstruct information that was never captured or has been destroyed.

## Features

| Capability | Method | Notes |
|---|---|---|
| Denoise | OpenCV non-local means | Reduces random scan and sensor noise |
| Deblock | Bilateral filtering | Approximation for mild compression artifacts; may soften detail |
| Classical inpaint | Telea / Navier–Stokes | Uses an explicit white-on-black repair mask |
| Enhance | Unsharp masking | Mild edge/detail enhancement |
| AI super-resolution | User-supplied ONNX model | Supports compatible one-input RGB image models |
| AI inpainting | User-supplied ONNX model | Supports compatible LaMa-style image-plus-mask exports |
| Quality metrics | PSNR and SSIM | Full-reference metrics when a clean image is available |
| Batch | CLI and web UI | Directory processing or multi-upload ZIP export |

Other features include grayscale/BGR/BGRA handling, PNG/JPEG/BMP/TIFF/WebP image I/O, alpha preservation, input validation, reproducible tests, and GitHub Actions CI.

## Install

Python 3.10+ is recommended.

~~~bash
git clone https://github.com/monika14r-m/Visual-Artifact-Removal.git
cd Visual-Artifact-Removal
python -m venv .venv
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

macOS / Linux:

~~~bash
source .venv/bin/activate
~~~

Install the core tools:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
~~~

Optional features are installed separately:

~~~bash
# ONNX inference (CPU)
pip install -r requirements-ai.txt

# Browser interface
pip install -r requirements-ui.txt
~~~

For NVIDIA GPU inference, use a compatible onnxruntime-gpu build and CUDA/cuDNN version for your system. The CLI can request the CUDA execution provider; CPU inference remains the default.

## Browser interface

Start the local workspace:

~~~bash
streamlit run app.py
~~~

The interface provides:
- Single-image processing with before/after comparison and PNG download.
- Mask upload for classical or AI inpainting.
- Multiple-image processing with per-file status and ZIP download.
- A model compatibility guide for ONNX inputs and outputs.

The batch tab currently supports the three classical operations. AI model inference is available in the single-image tab and through the CLI.

## CLI quick start

Denoise an image:

~~~bash
python -m src.main input.jpg output.png --operation denoise --strength 7
~~~

Reduce mild compression artifacts:

~~~bash
python -m src.main compressed.jpg restored.png --operation deblock --strength 5
~~~

Sharpen lightly:

~~~bash
python -m src.main soft.png enhanced.png --operation enhance --strength 7
~~~

Repair an explicitly selected region:

~~~bash
python -m src.main damaged.png repaired.png --operation inpaint --mask mask.png --method telea --radius 3
~~~

Run a compatible ONNX super-resolution model:

~~~bash
pip install -r requirements-ai.txt
python -m src.main low-resolution.png upscaled.png --operation ai-upscale --model models/upscaler.onnx
~~~

Run a LaMa-style two-input ONNX inpainting model:

~~~bash
python -m src.main damaged.png repaired.png --operation ai-inpaint --model models/lama.onnx --mask mask.png
~~~

Request CUDA inference when the matching runtime is installed:

~~~bash
python -m src.main input.png output.png --operation ai-upscale --model models/upscaler.onnx --device cuda
~~~

Batch process a directory:

~~~bash
python -m src.main --batch images/input images/output --operation denoise --strength 6 --recursive
~~~

Evaluate against a clean reference:

~~~bash
python -m src.main noisy.png restored.png --operation denoise --reference clean.png
~~~

PSNR and SSIM are only meaningful when the reference and output have matching dimensions and alignment.

## ONNX model compatibility

Model weights are not committed to this repository. They can be large, have separate licenses, and require model-specific preprocessing. The adapter intentionally supports a clear contract rather than claiming compatibility with every ONNX graph.

### Super-resolution / image restoration

The model must expose:
- Exactly one rank-4 image input and at least one output.
- Three RGB channels in NCHW or NHWC layout.
- Float32 input values normalized to [0, 1].
- Output image values in [0, 1], [-1, 1], or [0, 255].
- Dynamic spatial dimensions are recommended. Fixed dimensions are accepted only when the uploaded image matches them.

This can work with Real-ESRGAN-style ONNX exports when their exported tensor contract matches these requirements. Some exports use tiling, normalization, padding, or custom preprocessing and will need an adapter before they are compatible.

### LaMa-style inpainting

The model must expose exactly two rank-4 inputs: a three-channel RGB image and a one-channel mask. The mask input name should contain the word mask; otherwise the second input is treated as the mask. Both inputs use float32 values in [0, 1]. White mask pixels identify the area to reconstruct. The model output must match the input image dimensions.

Some LaMa exports have additional preprocessing, padding, or mask conventions. Check the model export details before use; matching the input names and dimensions alone does not prove equivalent results.

### Python API

~~~python
import cv2
from src.ai.onnx_restorer import ONNXRestorer

image = cv2.imread("input.png")
upscaler = ONNXRestorer("models/upscaler.onnx", task="upscale")
upscaled = upscaler.upscale(image)
cv2.imwrite("upscaled.png", upscaled)

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
inpainter = ONNXRestorer("models/lama.onnx", task="inpaint")
repaired = inpainter.inpaint(image, mask)
cv2.imwrite("repaired.png", repaired)
~~~

For a model whose output range cannot be inferred safely, pass output_range="zero_one", "minus_one_one", or "zero_255" to ONNXRestorer.

Only load model files from sources you trust. Model files are external inference inputs and can consume substantial memory or compute.

## Parameters and practical guidance

- Denoise strength: 0–30. Start around 5–8; higher values can erase texture.
- Deblock strength: 0–10. Start around 3–5; bilateral filtering can soften fine details.
- Inpainting radius: greater than 0 and at most 20. Small masks generally work better with smaller radii.
- Enhance strength on the CLI: 0–20 mapped to a restrained unsharp-mask amount.

Keep the original file and compare every result. AI model quality depends on its training data, preprocessing, input size, and export settings.

## Quality metrics

- **PSNR** measures pixel-level reconstruction error. Higher is generally better when comparing against a true clean reference.
- **SSIM** measures structural similarity. Values closer to 1 indicate greater structural similarity.

These are full-reference metrics. They cannot determine whether an output looks better when no clean reference exists, and they should not be used to compare images with different dimensions or alignment.

## Tests

~~~bash
pytest -q
~~~

Tests use synthetic images and fake ONNX sessions, so no model download or GPU is required for the test suite. GitHub Actions also runs a CLI smoke test and checks Python compilation for the web app.

## Limitations

- Classical filters cannot recover detail that has been completely destroyed by compression or blur.
- Deblocking is an edge-preserving smoothing approximation, not a learned JPEG restoration model.
- Classical inpainting quality depends on mask size and surrounding context.
- ONNX inference only supports the documented tensor contracts. It is not a universal ONNX model runner.
- Pretrained model weights are not bundled. The app does not automatically download or verify third-party weights.
- The browser batch queue currently runs classical operations; AI operations are available for individual images and via CLI.
- No video restoration or face-specific restoration is included.

## License

MIT. See the repository license file. Third-party ONNX model weights remain subject to their own licenses.
