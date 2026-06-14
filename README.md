Visual Artifact Removal

A Python toolkit for restoring degraded images — removing watermarks, compression artifacts, and visual defects using classical computer vision and inpainting techniques.

Overview
Visual Artifact Removal is a lightweight, extensible image restoration pipeline built on OpenCV. It is designed for researchers, developers, and hobbyists who need a reproducible, scriptable way to clean up images — without reaching for a heavy deep-learning stack.
The toolkit handles the three most common classes of visual degradation:
Artifact TypeMethod UsedWatermarksMask detection + inpaintingCompression artifactsBilateral filteringMinor defectsMorphological restoration
Project Structure
Visual-Artifact-Removal/
├── src/
│   ├── remove_watermark.py     # Core watermark removal pipeline
│   ├── artifact_filter.py      # Compression artifact reduction
│   └── utils.py                # Shared preprocessing utilities
├── tests/
│   ├── test_removal.py         # Integration tests
│   └── test_utils.py           # Unit tests for utility functions
├── requirements.txt
└── README.md
Installation
Requires Python 3.8+
bashgit clone https://github.com/monika14r-m/Visual-Artifact-Removal.git
cd Visual-Artifact-Removal
pip install -r requirements.txt
Usage
Remove a watermark
bashpython src/remove_watermark.py input.jpg output.jpg
Python API
pythonfrom src.remove_watermark import remove_watermark

remove_watermark(input_path="input.jpg", output_path="output.jpg")
Batch processing
pythonfrom src.utils import batch_process

batch_process(input_dir="images/raw/", output_dir="images/cleaned/")
Running Tests
bashpytest tests/ -v
All tests are written with pytest. The suite covers both unit-level utility functions and end-to-end removal pipelines.
How It Works

Preprocessing — The input image is normalized and converted to the appropriate color space using OpenCV.
Mask Generation — Artifact regions (e.g., watermark pixels) are detected via thresholding or edge analysis.
Inpainting — OpenCV's cv2.inpaint fills masked regions using the Navier-Stokes or Telea algorithm.
Post-processing — Output is sharpened and saved in the original format.

Roadmap

 Deep learning-based inpainting (e.g., LaMa, DeepFill)
 GUI interface for non-technical users
 Video artifact removal support
 Docker container for easy deployment

Contributing
Contributions are welcome. Please open an issue before submitting a pull request so we can discuss scope and approach.
bash# Fork the repo, create your branch
git checkout -b feature/your-feature-name

# Make changes, run tests
pytest tests/ -v

# Open a pull request
License
MIT License. See LICENSE for details.
