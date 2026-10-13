"""SAM2 automatic segmentation, mask filtering and item cropping."""
from dataclasses import dataclass

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_hf


@dataclass
class ItemCrop:
    mask: np.ndarray          # HxW bool, full image
    bbox: tuple               # (x0, y0, x1, y1) tight bbox, pixels
    bbox_crop: np.ndarray     # padded bbox crop (RGB), shown to the VLM
    masked_crop: np.ndarray   # tight crop with background set to white, used for embedding
    score: float              # SAM2 predicted IoU


class Segmenter:
    def __init__(self, cfg, device="cuda"):
        sam_cfg = dict(cfg)
        model_id = sam_cfg.pop("model_id")
        self.device = device
        self.model = build_sam2_hf(model_id, device=device)
        self.generator = SAM2AutomaticMaskGenerator(self.model, **sam_cfg)

    @torch.inference_mode()
    def generate(self, image):
        """image: HxWx3 uint8 RGB. Returns SAM2 raw mask dicts."""
        if self.device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.generator.generate(image)
        return self.generator.generate(image)


def _overlap(a, b):
    """Pixels shared by two SAM2 masks, counted only inside their bbox intersection."""
    ax, ay, aw, ah = (int(v) for v in a["bbox"])
    bx, by, bw, bh = (int(v) for v in b["bbox"])
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw) + 1, min(ay + ah, by + bh) + 1  # +1: bbox edges are inclusive
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
