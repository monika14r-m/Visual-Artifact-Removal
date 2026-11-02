"""Central project configuration defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Kept for compatibility with the original package API.
WATERMARK_THRESHOLD = 200
INPAINT_RADIUS = 3
RESIZE = False
RESIZE_WIDTH = 800
DEBUG = False
