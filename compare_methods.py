"""Compare the two recognition methods on photos whose contents you know.

    python compare_methods.py                    # the photos and contents listed in test/labels.yaml
    python compare_methods.py --labels my.yaml   # another list (photo paths relative to it)

The labels file lists what is really in each photo, by database folder name (unknown for an item that
is not in the database). Both methods run with the models in config.yaml and the app's default minimum
confidence. Each photo's results and annotated image are saved under outputs/compare/<method>/.
"""
import argparse
import gc
import json
import statistics
from collections import Counter
from pathlib import Path

import torch
import yaml

from svlm import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from run import process_image
from svlm.embed import Embedder, ItemIndex, build_index, index_dir_for, scan_db
from svlm.segment import Segmenter
from svlm.siglip import SiglipEmbedder, SiglipMatcher
from svlm.vlm import ItemRecognizer

METHODS = {"dino_qwen": "DINOv2 + Qwen", "siglip": "SigLIP"}


def load_method(method, cfg):
    """Search model id, embedder, recognizer, settings and minimum confidence, set up as in the app."""
    device = cfg["device"]
    if method == "siglip":
        embedder = SiglipEmbedder(cfg["siglip"]["model_id"], device=device)
        recognizer = SiglipMatcher(embedder, cfg["siglip"]["text_weight"])
        run_cfg = {**cfg, "retrieval": {**cfg["retrieval"], "top_k": cfg["siglip"]["top_k"]}}
        return cfg["siglip"]["model_id"], embedder, recognizer, run_cfg, cfg["siglip"]["min_score"]
    embedder = Embedder(cfg["embed"]["model_id"], device=device)
    recognizer = ItemRecognizer(cfg["vlm"]["model_id"], max_pixels=cfg["vlm"]["max_pixels"],
                                max_new_tokens=cfg["vlm"]["max_new_tokens"], device=device)
    return cfg["embed"]["model_id"], embedder, recognizer, cfg, 0.0


def load_index(embedder, model_id, cfg):
    """The search model's index, rebuilt first when it is missing or the database photos changed since
    it was built. (ItemIndex alone notices removed photos but not added ones.)"""
    index_dir, db_dir = index_dir_for(cfg["paths"]["index_dir"], model_id), cfg["paths"]["db_dir"]
    try:
        index = ItemIndex(index_dir, db_dir)
        if {str(Path(db_dir) / e["path"]) for e in scan_db(db_dir)} == {e["path"] for e in index.entries}:
            return index
    except (RuntimeError, OSError):
        pass
    print(f"Building the search index for {model_id} ...")
    build_index(embedder, db_dir, index_dir, batch_size=cfg["embed"]["batch_size"])
    return ItemIndex(index_dir, db_dir)


def score(expected, got):
    """expected, got: one photo's item names, "unknown" for an item not in the database.
    Returns (correct, wrong). wrong counts names given to something that is not that item:
    a wrong product, an item counted twice, or a piece of background named as a product."""
    e, g = Counter(expected), Counter(got)
    correct = sum((e & g).values())
    wrong = sum(n for item, n in (g - e).items() if item != "unknown")
    return correct, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(PROJECT_ROOT / "test" / "labels.yaml"))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = load_config(args.config)

    labels_path = Path(args.labels)
    photos = {labels_path.parent / name: sorted(items)
              for name, items in yaml.safe_load(labels_path.read_text(encoding="utf-8")).items()}
    missing = [str(p) for p in photos if not p.is_file()]
    if missing:
        ap.error(f"photos not found: {', '.join(missing)}")
    # A misspelt name would silently count as a miss
    names = {p.name for p in Path(cfg["paths"]["db_dir"]).iterdir() if p.is_dir()} | {"unknown"}
    typos = sorted({item for items in photos.values() for item in items} - names)
    if typos:
        ap.error(f"not database folder names: {', '.join(typos)}")
    n_items = sum(len(items) for items in photos.values())

    segmenter = Segmenter(cfg["sam2"], device=cfg["device"])
    out_root = Path(cfg["paths"]["out_dir"]) / "compare"
    rows, mistakes = [], []
    for method, name in METHODS.items():
        print(f"\n=== {name}: loading models ...")
        model_id, embedder, recognizer, run_cfg, min_confidence = load_method(method, cfg)
        index = load_index(embedder, model_id, cfg)
        torch.cuda.reset_peak_memory_stats()
        correct = wrong = perfect = 0
        times = []
        for photo, expected in photos.items():
            out_dir = out_root / method / photo.stem
            times.append(process_image(photo, out_dir, run_cfg, segmenter, embedder, index, recognizer,
                                       min_confidence)["total"])
            results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
            got = sorted(r["item"] or "unknown" for r in results)
            c, w = score(expected, got)
            correct, wrong, perfect = correct + c, wrong + w, perfect + (c == len(expected) and w == 0)
            if c < len(expected) or w:
                mistakes.append(f"  {name:<15}{photo.name}: expected {', '.join(expected)}"
                                f" | got {', '.join(got) or 'nothing'}")
        rows.append((name, correct, wrong, perfect, statistics.median(times),
                     torch.cuda.max_memory_allocated() / 2**30))
        # Free this method's models before loading the next, as the app does when switching
        del embedder, recognizer, index
        gc.collect()
        torch.cuda.empty_cache()

    models = {key: cfg[key]["model_id"].split("/")[-1] for key in ("sam2", "embed", "vlm", "siglip")}
    print(f"\n=== {len(photos)} photos, {n_items} items. SAM2: {models['sam2']}; "
          f"DINOv2 + Qwen: {models['embed']} + {models['vlm']}; SigLIP: {models['siglip']}")
    print(f"{'':<15}{'Found right':<14}{'Wrong/extra':<13}{'Photos fully right':<20}{'Median time':<13}Peak GPU")
    for name, correct, wrong, perfect, secs, gb in rows:
        print(f"{name:<15}{f'{correct}/{n_items} ({correct / n_items:.0%})':<14}{wrong:<13}"
              f"{f'{perfect}/{len(photos)}':<20}{f'{secs:.1f} s':<13}{gb:.1f} GB")
    print("\nMistakes:\n" + "\n".join(mistakes) if mistakes else "\nNo mistakes.")


if __name__ == "__main__":
    main()
