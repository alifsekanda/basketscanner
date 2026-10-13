import os
from pathlib import Path

import yaml
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# Keep model downloads (~20GB, mostly Qwen) on E: instead of the default C: cache.
# Must be set before transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models"))
# The models are already downloaded, so load them from there without contacting huggingface.co.
# To download a different model, run once with the environment variable HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_config(path=DEFAULT_CONFIG):
    """Relative entries under `paths:` are resolved against the config file's folder,
    so the scripts work no matter which directory they are started from."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str((path.parent / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def load_rgb(path):
    """Open an image as RGB with its EXIF orientation applied (phone photos are often
    stored sideways with an Orientation tag), and close the file."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
