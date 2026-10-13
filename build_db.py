"""Embed every image under db/<ItemName>/ and write the FAISS index.

    python build_db.py            # build index
    python build_db.py --eval     # build + leave-one-out retrieval accuracy
"""
import argparse
from collections import Counter

import numpy as np

from svlm import DEFAULT_CONFIG, load_config
from svlm.embed import Embedder, build_index, index_dir_for


def leave_one_out(entries, vecs, top_k):
    """For each db image, rank the other images' items; report top-1 / top-k accuracy."""
    labels = np.array([e["item"] for e in entries])
    counts = Counter(labels)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    top1 = topk = n = 0
    misses = []
    for i in range(len(entries)):
        if counts[labels[i]] < 2:  # no other image of this item to match against
            continue
        ranked_items = []
        for j in np.argsort(-sims[i]):
            if labels[j] not in ranked_items:
                ranked_items.append(labels[j])
            if len(ranked_items) == top_k:
                break
        n += 1
        top1 += ranked_items[0] == labels[i]
        topk += labels[i] in ranked_items
        if ranked_items[0] != labels[i]:
            misses.append((entries[i]["path"], ranked_items[0]))
    if n == 0:
        print("Leave-one-out needs at least 2 images per item.")
        return
    print(f"Leave-one-out over {n} images: top-1 {top1 / n:.1%}, top-{top_k} {topk / n:.1%}")
    for path, got in misses[:20]:
        print(f"  miss: {path} -> {got}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--eval", action="store_true", help="report leave-one-out retrieval accuracy")
    args = ap.parse_args()
    cfg = load_config(args.config)

    embedder = Embedder(cfg["embed"]["model_id"], device=cfg["device"])
    index_dir = index_dir_for(cfg["paths"]["index_dir"], cfg["embed"]["model_id"])
    entries, vecs = build_index(embedder, cfg["paths"]["db_dir"], index_dir,
                                batch_size=cfg["embed"]["batch_size"])
    n_items = len({e["item"] for e in entries})
    print(f"Indexed {len(entries)} images of {n_items} items -> {index_dir}")

    if args.eval:
        leave_one_out(entries, vecs, cfg["retrieval"]["top_k"])


if __name__ == "__main__":
    main()
