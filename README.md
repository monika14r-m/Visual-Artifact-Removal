# Visual Artifact Removal

A lightweight Python image-restoration toolkit for reducing common image noise and compression artifacts, repairing user-selected damaged regions, and evaluating restoration quality.

The project uses classical computer-vision methods with OpenCV. It is designed for transparent, reproducible processing—not as a promise to perfectly reconstruct lost image information.

## Features

| Operation | Method | Best for |
|---|---|---|
| Denoise | Non-local means | Random sensor / scan noise |
| Deblock | Bilateral filtering | Mild JPEG-like block and ringing artifacts |
| Inpaint | Telea or Navier–Stokes | Small damaged regions selected with a mask |
| Enhance | Unsharp masking | Mild edge/detail enhancement |
| Evaluate | PSNR and SSIM | Comparing output against a clean reference |
| Batch | Directory processing | Applying one operation to many images |

Other features:
- Grayscale, BGR and BGRA image support.
- PNG, JPEG, BMP, TIFF and WebP inputs through OpenCV.
- Explicit masks for inpainting; no automatic fixed-corner watermark guessing.
- CLI and Python API.
- Automated regression tests and GitHub Actions CI.

## Project structure

\`\`\`text
Visual-Artifact-Removal/
├── src/
│   ├── core/
│   │   ├── image_loader.py
│   │   ├── inpaint.py
│   │   ├── remover.py
│   │   └── utils.py
│   ├── config.py
│   ├── main.py
│   ├── metrics.py
│   └── removal_watermark.py
├── tests/
│   └── test_restoration.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
\`\`\`

## Installation

Python 3.10+ is recommended.

### Windows PowerShell

\`\`\`powershell
git clone https://github.com/monika14r-m/Visual-Artifact-Removal.git
cd Visual-Artifact-Removal
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
\`\`\`

### macOS / Linux

\`\`\`bash
git clone https://github.com/monika14r-m/Visual-Artifact-Removal.git
cd Visual-Artifact-Removal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
\`\`\`

## Quick start

Denoise a single image:

\`\`\`bash
python -m src.main input.jpg output.png --operation denoise --strength 7
\`\`\`

Reduce compression artifacts:

\`\`\`bash
python -m src.main compressed.jpg restored.png --operation deblock --strength 5
\`\`\`

Sharpen edges lightly:

\`\`\`bash
python -m src.main soft.png enhanced.png --operation enhance --strength 7
\`\`\`

Repair a selected region using a mask:

\`\`\`bash
python -m src.main damaged.png repaired.png --operation inpaint --mask mask.png --method telea --radius 3
\`\`\`

In a mask image, white pixels mark the region to repair and black pixels preserve the original image. The mask must have the same height and width as the source image.

Batch process a directory:

\`\`\`bash
python -m src.main --batch images/input images/output --operation denoise --strength 6 --recursive
\`\`\`

Evaluate against a clean reference:

\`\`\`bash
python -m src.main noisy.png restored.png --operation denoise --reference clean.png
\`\`\`

The CLI prints PSNR and SSIM when a reference image is supplied. The reference and restored output must have identical dimensions.

## Python API

\`\`\`python
import cv2

from src.core.image_loader import load_image, save_image
from src.core.remover import denoise_image, inpaint_image

image = load_image("input.jpg")
restored = denoise_image(image, strength=7)
save_image(restored, "output.png")

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
repaired = inpaint_image(image, mask, method="telea", radius=3)
save_image(repaired, "repaired.png")
\`\`\`

For a quality comparison:

\`\`\`python
from src.metrics import compare_quality

clean = load_image("clean_reference.png")
print(compare_quality(clean, restored))
\`\`\`

## Parameters and practical guidance

- Denoise strength: 0–30. Start around 5–8; higher values can erase texture.
- Deblock strength: 0–10. Start around 3–5; bilateral filtering can soften fine details.
- Inpainting radius: greater than 0 and at most 20. Small masks generally work better with smaller radii.
- Enhance strength: 0–20 on the CLI, mapped to a restrained unsharp-mask amount. Excessive sharpening amplifies noise and halos.

Always compare the output with the original. Keep the original file so restoration remains reversible.

## Quality metrics

- **PSNR** measures pixel-level reconstruction error. Higher is better when a true clean reference is available.
- **SSIM** measures structural similarity. Values closer to 1 indicate greater similarity.

These are full-reference metrics; they cannot determine whether an output looks better when no clean reference exists.

## Tests

\`\`\`bash
pytest -q
\`\`\`

Tests create synthetic images and cover file I/O, denoising, alpha-channel preservation, inpainting, parameter validation, resizing, and quality metrics. No sample image downloads are required.

## Limitations

- Classical filters cannot recover detail that has been completely destroyed by compression or blur.
- Deblocking is an edge-preserving smoothing approximation, not a learned JPEG restoration model.
- Inpainting quality depends on the mask and surrounding context; large masked regions may produce visible artifacts.
- Automatic watermark localization is intentionally not enabled. Supply a mask for regions you have the right to edit.
- This package does not currently provide deep-learning restoration, video processing, face restoration, or a GUI.

## License

MIT. See the repository license file.
