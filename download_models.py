"""Download every model listed under `models:` in config.yaml, only the files needed to run it.
Models already downloaded are checked and skipped.

    python download_models.py
"""
import os
from pathlib import Path
from urllib.request import urlretrieve

os.environ["HF_HUB_OFFLINE"] = "0"  # this script is the one place that goes online

from svlm import load_config  # noqa: E402  (also points HF_HOME at the project's models folder)

from huggingface_hub import snapshot_download  # noqa: E402

# FastSAM uses a .pt checkpoint; transformers models need safetensors + configs.
# max_workers=1: parallel downloads into this Windows cache can fail creating links.
PATTERNS = {"embed": {"ignore_patterns": ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx*"]},
            "vlm": {"ignore_patterns": ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx*"]},
            "siglip": {"ignore_patterns": ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx*"]}}

cfg = load_config()
for kind, model_ids in cfg["models"].items():
    for model_id in model_ids:
        print(f"{model_id} ...", flush=True)
        if kind == "fastsam":
            path = Path(__file__).resolve().parent / "models" / "fastsam" / model_id
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file() or path.stat().st_size == 0:
                download_path = path.with_suffix(path.suffix + ".download")
                try:
                    urlretrieve(
                        f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_id}",
                        download_path,
                    )
                    download_path.replace(path)
                finally:
                    download_path.unlink(missing_ok=True)
        else:
            path = snapshot_download(model_id, max_workers=1, **PATTERNS[kind])
        print(f"  ok -> {path}", flush=True)
print("All models downloaded.")
