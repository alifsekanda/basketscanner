"""FastSAM automatic segmentation, mask filtering and item cropping."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ItemCrop:
    mask: np.ndarray          # HxW bool, full image
    bbox: tuple               # (x0, y0, x1, y1) tight bbox, pixels
    bbox_crop: np.ndarray     # padded bbox crop (RGB), shown to the VLM
    masked_crop: np.ndarray   # tight crop with background set to white, used for embedding
    score: float              # FastSAM detection confidence


class Segmenter:
    def __init__(self, cfg, device="cuda"):
        self.device = device
        self.imgsz = cfg["imgsz"]
        self.conf = cfg["conf"]
        self.iou = cfg["iou"]
        self.max_det = cfg["max_det"]

        model_id = cfg["model_id"]
        model_path = Path(model_id)
        if not model_path.is_file():
            model_path = Path(__file__).resolve().parent.parent / "models" / "fastsam" / model_id
        if not model_path.is_file():
            raise FileNotFoundError(
                f"FastSAM weights not found: {model_id}. Run `python download_models.py` first.")

        from ultralytics import FastSAM

        self.model = FastSAM(str(model_path))

    def generate(self, image):
        """image: HxWx3 uint8 RGB. Returns masks in the shared SAM-style format."""
        device = self.device
        if device == "cuda":
            device = "0"
        elif device.startswith("cuda:"):
            device = device.split(":", 1)[1]

        results = self.model.predict(
            source=image[:, :, ::-1].copy(),
            device=device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            retina_masks=True,
            verbose=False,
        )
        if not results or results[0].masks is None:
            return []

        result = results[0]
        masks = result.masks.data.cpu().numpy()
        scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                  else np.ones(len(masks), dtype=np.float32))
        generated = []
        for mask, score in zip(masks, scores):
            segmentation = mask.astype(bool)
            ys, xs = np.nonzero(segmentation)
            if not len(xs):
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            generated.append({
                "segmentation": segmentation,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": int(segmentation.sum()),
                "predicted_iou": float(score),
            })
        return generated


def _overlap(a, b):
    """Pixels shared by two masks, counted only inside their half-open bbox intersection."""
    ax, ay, aw, ah = (int(v) for v in a["bbox"])
    bx, by, bw, bh = (int(v) for v in b["bbox"])
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x0 >= x1 or y0 >= y1:
        return 0
    return np.logical_and(a["segmentation"][y0:y1, x0:x1], b["segmentation"][y0:y1, x0:x1]).sum()


def filter_masks(masks, image_shape, min_area_frac, max_area_frac, max_bbox_frac, containment_thresh):
    """Keep one mask per item: drop tiny parts and huge basket/background masks
    (by pixel area, and by bbox area to catch thin rings like the basket rim),
    then drop any mask that mostly lies inside a larger kept mask (labels, caps...)."""
    h, w = image_shape[:2]
    total = h * w
    candidates = [m for m in masks
                  if min_area_frac * total <= m["area"] <= max_area_frac * total
                  and m["bbox"][2] * m["bbox"][3] <= max_bbox_frac * total]
    candidates.sort(key=lambda m: m["area"], reverse=True)

    kept = []
    for m in candidates:
        inside = any(_overlap(m, k) / max(m["area"], 1) >= containment_thresh for k in kept)
        if not inside:
            kept.append(m)
    return kept


def crop_items(image, masks, pad_frac):
    h, w = image.shape[:2]
    items = []
    for m in masks:
        seg = m["segmentation"].astype(bool)
        ys, xs = np.nonzero(seg)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1

        pad = int(pad_frac * max(x1 - x0, y1 - y0))
        px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
        px1, py1 = min(w, x1 + pad), min(h, y1 + pad)

        masked = np.full_like(image[y0:y1, x0:x1], 255)
        local = seg[y0:y1, x0:x1]
        masked[local] = image[y0:y1, x0:x1][local]

        items.append(ItemCrop(
            mask=seg,
            bbox=(int(x0), int(y0), int(x1), int(y1)),
            bbox_crop=image[py0:py1, px0:px1].copy(),
            masked_crop=masked,
            score=float(m["predicted_iou"]),
        ))
    # Stable reading order: top-to-bottom, then left-to-right
    items.sort(key=lambda it: (it.bbox[1] // 50, it.bbox[0]))
    return items
