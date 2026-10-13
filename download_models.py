"""Download every model listed under `models:` in config.yaml, only the files needed to run it.
Models already downloaded are checked and skipped.

    python download_models.py
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"  # this script is the one place that goes online

from svlm import load_config  # noqa: E402  (also points HF_HOME at the project's models folder)

from huggingface_hub import snapshot_download  # noqa: E402

# sam2 builds its model from the .pt checkpoint; transformers models need safetensors + configs.
# max_workers=1: parallel downloads into this Windows cache can fail creating links.
PATTERNS = {"sam2": {"allow_patterns": ["*.pt"]},
            "embed": {"ignore_patterns": ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx*"]},
            "vlm": {"ignore_patterns": ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx*"]},
            "siglip": {"ignore_patterns": ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx*"]}}

cfg = load_config()
for kind, model_ids in cfg["models"].items():
    for model_id in model_ids:
        print(f"{model_id} ...", flush=True)
        path = snapshot_download(model_id, max_workers=1, **PATTERNS[kind])
        print(f"  ok -> {path}", flush=True)
print("All models downloaded.")
