# Visual Artifact Removal

A Python-based image restoration toolkit for removing watermarks, reducing compression artifacts, and repairing minor visual defects using OpenCV and classical computer vision techniques.

---

## Overview

Visual Artifact Removal is a lightweight image-processing project that demonstrates practical applications of image restoration techniques using OpenCV.

The toolkit provides a modular pipeline for detecting and repairing common visual artifacts while remaining easy to understand, extend, and integrate into larger computer vision workflows.

### Supported Operations

| Artifact              | Technique                 |
| --------------------- | ------------------------- |
| Watermarks            | Mask-based inpainting     |
| Compression Artifacts | Bilateral filtering       |
| Minor Defects         | Morphological restoration |

---

## Key Features

* Watermark removal using OpenCV inpainting
* Compression artifact reduction
* Morphological image restoration
* Batch image processing support
* Modular and extensible architecture
* Automated testing with PyTest
* Simple Python API and CLI usage

---

## Project Structure

```text
Visual-Artifact-Removal/
├── src/
│   ├── remove_watermark.py
│   ├── artifact_filter.py
│   └── utils.py
│
├── tests/
│   ├── test_removal.py
│   └── test_utils.py
│
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

* Python 3.8+
* OpenCV
* NumPy
* PyTest

### Setup

```bash
git clone https://github.com/monika14r-m/Visual-Artifact-Removal.git

cd Visual-Artifact-Removal

pip install -r requirements.txt
```

---

## Usage

### Command Line

```bash
python src/remove_watermark.py input.jpg output.jpg
```

### Python API

```python
from src.remove_watermark import remove_watermark

remove_watermark(
    input_path="input.jpg",
    output_path="output.jpg"
)
```

### Batch Processing

```python
from src.utils import batch_process

batch_process(
    input_dir="images/raw/",
    output_dir="images/cleaned/"
)
```

---

## Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

The project includes both unit tests and integration tests to validate image restoration workflows.

---

## Processing Pipeline

### 1. Preprocessing

Images are normalized and prepared for analysis using OpenCV.

### 2. Artifact Detection

Target regions are identified using thresholding, masking, and edge-based analysis techniques.

### 3. Restoration

Detected artifacts are repaired using OpenCV inpainting algorithms, including:

* Telea Inpainting
* Navier-Stokes Inpainting

### 4. Post-Processing

The restored image is refined and exported in its original format.

---

## Future Work

* Deep-learning-based inpainting (LaMa, DeepFill)
* Video restoration support
* Interactive desktop GUI
* Dockerized deployment
* Automatic watermark localization

---

## Contributing

Contributions are welcome.

```bash
git checkout -b feature/new-feature

pytest tests/ -v

git commit -m "Add new feature"
```

Please open an issue before submitting major changes.

---

## License

Licensed under the MIT License.

See the LICENSE file for details.
