"""DINOv2 embeddings and a FAISS index over the item database."""
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from svlm import load_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Embedder:
    def __init__(self, model_id, device="cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.input_size = self.processor.crop_size["height"]

    @torch.inference_mode()
    def embed(self, images, batch_size=32):
        """images: list of file paths, PIL images or HxWx3 uint8 RGB arrays.
        Returns (N, D) L2-normalized float32."""
        out = []
        for i in range(0, len(images), batch_size):
            # The processor's default resize + center crop would cut the ends off long thin
            # items (pens, bottles), so pad to a square and resize the whole image instead.
            batch = [pad_to_square(_to_pil(im)).resize((self.input_size, self.input_size),
                                                       Image.BICUBIC)
                     for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, do_resize=False, do_center_crop=False,
                                    return_tensors="pt").to(self.device)
            feats = self.model(**inputs).pooler_output
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            out.append(feats.cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def _to_pil(im):
    if isinstance(im, (str, Path)):
        return load_rgb(im)
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    return Image.fromarray(im).convert("RGB")


def pad_to_square(im, fill=(255, 255, 255)):
    w, h = im.size
    if w == h:
        return im
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def scan_db(db_dir):
    """db/<ItemName>/*.jpg -> list of {"item", "path"}, path relative to db_dir."""
    db_dir = Path(db_dir)
    if not db_dir.is_dir():
        raise FileNotFoundError(f"Database folder not found: {db_dir}")
    entries = []
    for item_dir in sorted(p for p in db_dir.iterdir() if p.is_dir()):
        for img in sorted(item_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                entries.append({"item": item_dir.name, "path": img.relative_to(db_dir).as_posix()})
    return entries


def index_dir_for(index_root, model_id):
    """Each embedding model gets its own index: vectors from different models can't be compared."""
    return Path(index_root) / model_id.split("/")[-1]


def build_index(embedder, db_dir, index_dir, batch_size=32):
    entries = scan_db(db_dir)
    if not entries:
        raise ValueError(f"No images found under {db_dir}/<ItemName>/")
    vecs = embedder.embed([Path(db_dir) / e["path"] for e in entries], batch_size=batch_size)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entries, vecs


class ItemIndex:
    def __init__(self, index_dir, db_dir):
        index_dir, db_dir = Path(index_dir), Path(db_dir)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with open(index_dir / "labels.json", encoding="utf-8") as f:
            self.entries = json.load(f)
        for e in self.entries:
            e["path"] = str(db_dir / e["path"])

        # Fail now, before the VLM loads, if the index is stale or from an older version.
        missing = [e["path"] for e in self.entries if not Path(e["path"]).is_file()]
        if missing or self.index.ntotal != len(self.entries):
            detail = f"{len(missing)} reference images are missing (e.g. {missing[0]})" if missing \
                else "index.faiss and labels.json disagree"
            raise RuntimeError(f"Index in {index_dir} is out of date: {detail}. Re-run build_db.py.")

    def search(self, vec, top_k=5):
        """Returns up to top_k distinct items, best first:
        [{"item", "score", "path"}] where path is that item's best-matching reference image.
        Searches every image (exact and cheap for a flat index), so an item with many
        reference photos cannot crowd the other items out of the results."""
        scores, ids = self.index.search(vec.reshape(1, -1).astype(np.float32), self.index.ntotal)
        best = {}
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            e = self.entries[idx]
            if e["item"] not in best:  # results are sorted, so first hit is the best
                best[e["item"]] = {"item": e["item"], "score": float(score), "path": e["path"]}
            if len(best) == top_k:
                break
        return list(best.values())
