"""Segment a basket photo, crop each item and identify it against the database.

    python run.py basket.jpg [more.jpg ...] [--no-vlm]
    python run.py photos\\*.jpg
    python run.py photos            # every image in the folder
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from svlm import DEFAULT_CONFIG, load_config, load_rgb
from svlm.embed import IMAGE_EXTS, Embedder, ItemIndex, index_dir_for
from svlm.segment import Segmenter, crop_items, filter_masks

FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(image, items, results):
    """Draws on the RGB image directly (cv2 drawing doesn't care about channel order).
    Lines and labels grow with the photo, so they stay readable when it is shown scaled down."""
    vis = image.copy()
    h, w = vis.shape[:2]
    s = max(1.0, max(h, w) / 600)
    font_scale, thick, pad = 0.6 * s, max(2, round(s)), round(4 * s)
    rng = np.random.default_rng(0)
    for it in items:
        color = rng.integers(60, 255, 3)
        vis[it.mask] = (0.5 * vis[it.mask] + 0.5 * color).astype(np.uint8)
    for it, res in zip(items, results):
        x0, y0, x1, y1 = it.bbox
        label = f"{res['id']}: {res['item'] or 'unknown'}"
        if res["confidence"] is not None:
            label += f" ({res['confidence']:.2f})"
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), thick)
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thick)
        # Label sits above the box, or just inside it when the box touches the top edge
        ly = y0 - th - 2 * pad if y0 - th - 2 * pad >= 0 else y0
        lx = max(0, min(x0, w - tw - pad))
        cv2.rectangle(vis, (lx, ly), (lx + tw + pad, ly + th + 2 * pad), (0, 255, 0), -1)
        cv2.putText(vis, label, (lx + pad // 2, ly + th + pad), FONT, font_scale, (0, 0, 0), thick)
    return vis


def expand_inputs(args):
    """Windows shells don't expand wildcards, so do it here; folders expand to their images.
    Wildcards and folders only pick up image files; duplicates are dropped."""
    paths = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif glob.has_magic(a):
            paths += sorted(Path(f) for f in glob.glob(a) if Path(f).suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(p)
    seen, unique = set(), []
    for p in paths:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def output_dir(out_root, img_path, used):
    """outputs/<stem>, with _2, _3... when two inputs in this run share a stem
    (a/img1.jpg + b/img1.jpg, or img1.jpg + img1.png). Case-insensitive, as on Windows."""
    name, n = img_path.stem, 2
    while name.lower() in used:
        name, n = f"{img_path.stem}_{n}", n + 1
    used.add(name.lower())
    return out_root / name


def format_times(times):
    return " · ".join(f"{step} {secs:.1f} s" for step, secs in times.items())


def measure_mm(mask, micron_per_pixel):
    """(diameter, length) in mm: the short and long side of the tightest rotated rectangle
    around the mask, so it works whatever way the item lies."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return min(w, h) * micron_per_pixel / 1000, max(w, h) * micron_per_pixel / 1000


def battery_size(decision, mask, cfg):
    """For an item identified as a battery, the measured diameter picks the size (config battery_size),
    overriding the VLM. Returns the decision and the measured (diameter, length) in mm, or None."""
    bs = cfg.get("battery_size") or {}
    if not bs.get("micron_per_pixel") or decision["item"] not in bs["diameters_mm"]:
        return decision, None
    size_mm = measure_mm(mask, bs["micron_per_pixel"])
    size, diameter = min(bs["diameters_mm"].items(), key=lambda kv: abs(kv[1] - size_mm[0]))
    measured = f"measured {size_mm[0]:.1f} x {size_mm[1]:.1f} mm"
    if abs(diameter - size_mm[0]) > bs["tolerance_mm"]:
        return {**decision, "item": None,
                "reason": f"{measured}, no battery size matches (guess: {decision['item']})"}, size_mm
    note = "size confirmed" if size == decision["item"] else f"size corrected from {decision['item']}"
    return {**decision, "item": size, "reason": f"{measured}, {note}"}, size_mm


def process_image(img_path, out_dir, cfg, segmenter, embedder, index, recognizer, min_confidence=0.0):
    """Items whose confidence is below min_confidence are reported as unknown.
    Returns how long each step took, in seconds."""
    t0 = time.perf_counter()
    image = np.array(load_rgb(img_path))
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("item_*.png"):  # leftovers from an earlier run of this image
        old.unlink()

    raw = segmenter.generate(image)
    f = cfg["filter"]
    kept = filter_masks(raw, image.shape, f["min_area_frac"], f["max_area_frac"],
                        f["max_bbox_frac"], f["containment_thresh"])
    items = crop_items(image, kept, f["bbox_pad_frac"])
    print(f"{img_path}: {len(raw)} raw masks -> {len(items)} items")
    t1 = time.perf_counter()

    vecs = embedder.embed([it.masked_crop for it in items]) if items else []
    all_candidates = [index.search(vec, top_k=cfg["retrieval"]["top_k"]) for vec in vecs]
    t2 = time.perf_counter()
    # One batched VLM call for all crops is much faster than one call per crop
    decisions = recognizer.identify([it.bbox_crop for it in items], all_candidates) if recognizer else None
    t3 = time.perf_counter()
    results = []
    for i, (it, candidates) in enumerate(zip(items, all_candidates)):
        Image.fromarray(it.bbox_crop).save(crops_dir / f"item_{i:02d}_bbox.png")
        Image.fromarray(it.masked_crop).save(crops_dir / f"item_{i:02d}_masked.png")

        if recognizer:
            decision = decisions[i]
        else:
            top = candidates[0] if candidates else None
            decision = {"item": top and top["item"], "confidence": top and top["score"],
                        "reason": "retrieval top-1", "raw": ""}
        decision, size_mm = battery_size(decision, it.mask, cfg)
        if decision["item"] and (decision["confidence"] or 0) < min_confidence:
            decision = {**decision, "item": None,
                        "reason": f"below minimum confidence {min_confidence:.2f} (guess: {decision['item']})"}

        res = {"id": i, "bbox": list(it.bbox), "item": decision["item"],
               "confidence": decision["confidence"], "reason": decision["reason"],
               "vlm_raw": decision["raw"], "sam_score": it.score, "candidates": candidates,
               "size_mm": [round(v, 1) for v in size_mm] if size_mm else None}
        results.append(res)
        print(f"  [{i:02d}] {res['item'] or 'unknown':<25} conf={res['confidence']}  "
              f"retrieval top-1={candidates[0]['item'] if candidates else '-'}"
              + (f"  size {res['size_mm'][0]} x {res['size_mm'][1]} mm" if size_mm else ""))

    with open(out_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    # PIL rather than cv2.imwrite, which silently fails on non-ASCII paths on Windows
    Image.fromarray(annotate(image, items, results)).save(out_dir / "annotated.jpg", quality=95)

    times = {"segmentation": t1 - t0, "search": t2 - t1}
    if recognizer:
        times[getattr(recognizer, "step_name", "VLM")] = t3 - t2
    times["total"] = time.perf_counter() - t0
    print(f"  time: {format_times(times)}")
    print(f"  -> {out_dir}")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-vlm", action="store_true", help="use retrieval top-1 only (skip Qwen)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = cfg["device"]
    out_root = Path(args.out or cfg["paths"]["out_dir"])
    images = expand_inputs(args.images)
    if not images:
        ap.error(f"no images found for {args.images}")

    print("Loading DINOv2 + index...")
    index = ItemIndex(index_dir_for(cfg["paths"]["index_dir"], cfg["embed"]["model_id"]), cfg["paths"]["db_dir"])
    embedder = Embedder(cfg["embed"]["model_id"], device=device)
    print("Loading SAM2...")
    segmenter = Segmenter(cfg["sam2"], device=device)

    recognizer = None
    if not args.no_vlm:
        print("Loading Qwen2.5-VL...")
        from svlm.vlm import ItemRecognizer
        recognizer = ItemRecognizer(cfg["vlm"]["model_id"], max_pixels=cfg["vlm"]["max_pixels"],
                                    max_new_tokens=cfg["vlm"]["max_new_tokens"], device=device)

    used, failed = set(), []
    for img_path in images:
        try:
            process_image(img_path, output_dir(out_root, img_path, used), cfg,
                          segmenter, embedder, index, recognizer)
        except Exception as e:  # one bad file shouldn't abort the whole batch
            print(f"{img_path}: FAILED - {type(e).__name__}: {e}")
            failed.append(img_path)

    if failed:
        print(f"\n{len(failed)} of {len(images)} images failed: {', '.join(map(str, failed))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
