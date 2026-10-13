"""Local web UI: pick a recognition method on the home page, then identify the items in basket photos,
compare models, and rebuild the item index.

    python app.py      # opens http://127.0.0.1:7860; a method's models load when you start it
"""
# svlm must be imported before gradio: it points HF_HOME at the E: model cache, and
# gradio imports huggingface_hub, which reads HF_HOME once at import time.
from svlm import load_config

import base64
import contextlib
import gc
import html
import io
import json
import time
from collections import Counter
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from build_db import leave_one_out
from run import process_image
from svlm.embed import Embedder, ItemIndex, build_index, index_dir_for, scan_db
from svlm.segment import Segmenter
from svlm.siglip import SiglipEmbedder, SiglipMatcher
from svlm.vlm import ItemRecognizer

cfg = load_config()
paths, device = cfg["paths"], cfg["device"]

# One card per method on the home page, in this order.
METHODS = {
    "siglip": {
        "name": "SigLIP", "kinds": ("sam2", "siglip"), "recommended": True,
        "steps": "SAM2 → SigLIP image and name match",
        "about": "Matches each item against your product photos and product names. Fast and light, and it "
                 "named the most items correctly in our test.",
        "facts": (("Per photo", "1.9 s"), ("GPU memory", "5.2 GB"), ("Items right", "13 of 14")),
    },
    "dino_qwen": {
        "name": "DINOv2 + Qwen", "kinds": ("sam2", "embed", "vlm"), "recommended": False,
        "steps": "SAM2 → DINOv2 shortlist → Qwen VLM",
        "about": "DINOv2 shortlists the 5 closest products, then the Qwen vision-language model picks one "
                 "or answers unknown. Slower, and needs a 24 GB GPU.",
        "facts": (("Per photo", "5.5 s"), ("GPU memory", "18.9 GB"), ("Items right", "11 of 14")),
    },
}
LOADERS = {
    "sam2": lambda model_id: Segmenter({**cfg["sam2"], "model_id": model_id}, device=device),
    "embed": lambda model_id: Embedder(model_id, device=device),
    "vlm": lambda model_id: ItemRecognizer(model_id, max_pixels=cfg["vlm"]["max_pixels"],
                                           max_new_tokens=cfg["vlm"]["max_new_tokens"], device=device),
    "siglip": lambda model_id: SiglipEmbedder(model_id, device=device),
}
loaded = {}   # kind -> (model_id, model): one model of each kind on the GPU at a time
indexes = {}  # search model_id -> ItemIndex

THEME = gr.themes.Soft(
    primary_hue="teal", neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"])
# Colours come from the theme's variables, so the custom parts follow light and dark mode.
CSS = """
.bs-header { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 16px; }
.bs-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
.bs-header p { margin: 0; color: var(--body-text-color-subdued); }
.bs-note { margin: 0; font-size: 0.8rem; color: var(--body-text-color-subdued); }

.method-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 20px !important;
               background: var(--block-background-fill); justify-content: space-between; }
.method-card.recommended { border-color: var(--color-accent); box-shadow: 0 0 0 1px var(--color-accent); }
.mc-badge { display: inline-block; margin-bottom: 8px; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-accent);
            background: var(--color-accent-soft); }
.mc-name { margin: 0; font-size: 1.4rem; font-weight: 700; }
.mc-steps { margin: 2px 0 12px; font-family: var(--font-mono); font-size: 0.8rem; color: var(--body-text-color-subdued); }
.mc-about { margin: 0 0 16px; }
.mc-facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0; }
.mc-facts div { padding: 8px 10px; border-radius: 10px; background: var(--background-fill-secondary); }
.mc-facts dt { font-size: 0.72rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--body-text-color-subdued); }
.mc-facts dd { margin: 0; font-size: 1.05rem; font-weight: 600; font-variant-numeric: tabular-nums; }

.bs-method { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; }
.bs-method-name { font-size: 1.2rem; font-weight: 700; }
.bs-method-steps { font-family: var(--font-mono); font-size: 0.8rem; color: var(--body-text-color-subdued); }

.bs-empty { margin: 0; padding: 12px 2px; color: var(--body-text-color-subdued); }
.bs-stats { display: flex; flex-wrap: wrap; gap: 10px; }
.bs-stat { min-width: 110px; padding: 8px 14px; border-radius: 10px; background: var(--background-fill-secondary); }
.bs-stat .v { display: block; font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.bs-stat .k { font-size: 0.78rem; color: var(--body-text-color-subdued); }
.bs-contents { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.bs-chip { padding: 2px 10px; border: 1px solid var(--border-color-primary); border-radius: 999px; font-size: 0.85rem; }
.bs-chip.unknown { border-style: dashed; color: var(--body-text-color-subdued); }
.bs-foot { margin: 8px 0 0; font-size: 0.78rem; color: var(--body-text-color-subdued); overflow-wrap: anywhere; }
.bs-warn { margin: 10px 0 0; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--color-accent);
           background: var(--color-accent-soft); }

.bs-items { display: grid; gap: 8px; }
.bs-item { display: grid; grid-template-columns: 22px 64px minmax(0, 1fr) 84px; align-items: center; gap: 10px;
           padding: 8px; border: 1px solid var(--border-color-primary); border-radius: 10px;
           background: var(--block-background-fill); }
.bs-item.unknown { border-style: dashed; }
.bs-num { font-family: var(--font-mono); font-size: 0.8rem; text-align: center; color: var(--body-text-color-subdued); }
.bs-thumb { width: 64px; height: 64px; object-fit: contain; border-radius: 6px; background: var(--background-fill-secondary); }
.bs-name { font-weight: 600; overflow-wrap: anywhere; }
.bs-item.unknown .bs-name { font-style: italic; color: var(--body-text-color-subdued); }
.bs-sub { font-size: 0.8rem; color: var(--body-text-color-subdued); overflow-wrap: anywhere; }
.bs-conf { display: grid; gap: 4px; justify-items: end; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.bs-bar { width: 100%; height: 6px; overflow: hidden; border-radius: 3px; background: var(--background-fill-secondary); }
.bs-bar span { display: block; height: 100%; background: var(--color-accent); }
.bs-item.unknown .bs-bar span { background: var(--body-text-color-subdued); }
"""


def get_model(kind, model_id, on_load=None):
    """The requested model, loading it if needed. The model it replaces is freed first,
    so switching never needs room for both on the GPU. on_load(model_id) is called before a load
    (used to show which model is loading in the page)."""
    if kind in loaded and loaded[kind][0] == model_id:
        return loaded[kind][1]
    loaded.pop(kind, None)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Loading {model_id} ...")
    if on_load:
        on_load(model_id)
    loaded[kind] = (model_id, LOADERS[kind](model_id))
    return loaded[kind][1]


def short(model_id):
    return model_id.split("/")[-1]


def free_unused(kinds):
    """Free the GPU memory of models the chosen method does not use."""
    for kind in [k for k in loaded if k not in kinds]:
        loaded.pop(kind)
    gc.collect()
    torch.cuda.empty_cache()


def search_kind(model_id):
    return "siglip" if model_id in cfg["models"]["siglip"] else "embed"


def load_index(model_id):
    """This search model's saved index, or None when there is none (or it lists photos that are gone)."""
    if model_id not in indexes:
        try:
            indexes[model_id] = ItemIndex(index_dir_for(paths["index_dir"], model_id), paths["db_dir"])
        except (RuntimeError, OSError):
            return None
    return indexes[model_id]


def index_changes(index, entries=None):
    """(added, removed): database photos the index lacks, and indexed photos that are gone.
    ItemIndex itself only notices removed photos."""
    entries = scan_db(paths["db_dir"]) if entries is None else entries
    on_disk = {str(Path(paths["db_dir"]) / e["path"]) for e in entries}
    indexed = {e["path"] for e in index.entries}
    return len(on_disk - indexed), len(indexed - on_disk)


def build(model_id, progress=None, changed=False):
    """Build this search model's index from the database folder. Returns entries and vectors for the
    accuracy check. changed: an older index exists (only changes the progress message)."""
    # (a numeric value is needed for Gradio to show the text: 0 = text and an empty bar).
    # "is not None": a gr.Progress has len() 0, so it is falsy
    say = (lambda desc: progress(0, desc=desc)) if progress is not None else (lambda desc: None)
    embedder = get_model(search_kind(model_id), model_id, on_load=lambda m: say(f"Loading {short(m)} ..."))
    say(f"The database changed: updating the search index for {short(model_id)} ..." if changed else
        f"Building the search index for {short(model_id)} (once per search model) ...")
    entries, vecs = build_index(embedder, paths["db_dir"], index_dir_for(paths["index_dir"], model_id),
                                batch_size=cfg["embed"]["batch_size"])
    indexes.pop(model_id, None)
    return entries, vecs


def thumb_uri(path, size=128):
    """A crop as a small inline JPEG, so the item list needs no file serving."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def summary_html(results, times, load_secs, models, peak_gb, memory_full, out_dir):
    counts = Counter(r["item"] or "unknown" for r in results)
    n_unknown = counts.pop("unknown", 0)
    steps = " · ".join(f"{step} {secs:.1f} s" for step, secs in times.items() if step != "total")
    stats = [(len(results), "items found"), (n_unknown, "unknown"), (f"{times['total']:.1f} s", steps),
             (f"{peak_gb:.1f} GB", "peak GPU memory")]
    out = ['<div class="bs-stats">'] + [f'<div class="bs-stat"><span class="v">{v}</span>'
                                        f'<span class="k">{html.escape(k)}</span></div>' for v, k in stats] + ["</div>"]
    chips = [f'<span class="bs-chip">{html.escape(item)} × {n}</span>' for item, n in counts.most_common()]
    if n_unknown:
        chips.append(f'<span class="bs-chip unknown">unknown × {n_unknown}</span>')
    if chips:
        out.append(f'<div class="bs-contents">{"".join(chips)}</div>')
    else:
        out.append('<p class="bs-empty">No items found. Try a clearer top-down photo, or adjust '
                   '<code>filter:</code> in config.yaml.</p>')
    foot = "Models: " + " · ".join(short(m) for m in models)
    if load_secs > 1:
        foot += f" · loading took {load_secs:.0f} s (not counted)"
    out.append(f'<p class="bs-foot">{html.escape(foot)}<br>Saved to {html.escape(str(out_dir))}</p>')
    if memory_full:
        out.append('<p class="bs-warn"><b>GPU memory is full.</b> Windows then moves data into normal RAM, which '
                   'makes everything much slower, so these times are not meaningful. Pick smaller models '
                   '(DINOv2-giant with a 7B/8B VLM does not fit in 24 GB).</p>')
    return "".join(out)


def items_html(results, out_dir):
    """One row per item: its number on the result image, the crop, the name and the confidence."""
    rows = []
    for r in results:
        top = r["candidates"][0] if r["candidates"] else None
        match = f"{top['item']} ({top['score']:.2f})" if top else "-"
        if r["item"] is None:
            name, note = "Unknown", f"closest: {match}"
        elif r.get("size_mm") or r["reason"] not in ("", "retrieval top-1"):
            name, note = r["item"], r["reason"]  # SigLIP's image/name scores, or the battery size check
        else:
            name, note = r["item"], f"search match: {match}"
        conf = r["confidence"] or 0.0
        thumb = thumb_uri(out_dir / "crops" / f"item_{r['id']:02d}_bbox.png")
        rows.append(
            f'<div class="bs-item{" unknown" if r["item"] is None else ""}">'
            f'<span class="bs-num">{r["id"]}</span><img class="bs-thumb" src="{thumb}" alt="">'
            f'<div><div class="bs-name">{html.escape(name)}</div><div class="bs-sub">{html.escape(note)}</div></div>'
            f'<div class="bs-conf"><span>{conf:.2f}</span>'
            f'<div class="bs-bar"><span style="width: {min(conf, 1.0) * 100:.0f}%"></span></div></div></div>')
    return f'<div class="bs-items">{"".join(rows)}</div>'


def identify(photo, method, use_vlm, sam_id, search_id, vlm_id, min_confidence, progress=None):
    if photo is None:
        raise gr.Error("Add a basket photo first.")
    t = time.perf_counter()
    loading = (lambda m: progress(0, desc=f"Loading {short(m)} ...")) if progress is not None else None
    segmenter = get_model("sam2", sam_id, loading)
    embedder = get_model(search_kind(search_id), search_id, loading)
    if method == "siglip":
        recognizer = SiglipMatcher(embedder, cfg["siglip"]["text_weight"])
        run_cfg = {**cfg, "retrieval": {**cfg["retrieval"], "top_k": cfg["siglip"]["top_k"]}}
        models = [sam_id, search_id]
    else:
        recognizer = get_model("vlm", vlm_id, loading) if use_vlm else None
        run_cfg = cfg
        models = [sam_id, search_id] + ([vlm_id] if use_vlm else [])
    # Build the index on the first use of a search model, and rebuild it when photos were added or removed:
    # an index without the new photos would silently never find those products
    index = load_index(search_id)
    if index is None or any(index_changes(index)):
        build(search_id, progress, changed=index_dir_for(paths["index_dir"], search_id).exists())
        index = load_index(search_id)
    load_secs = time.perf_counter() - t
    if progress is not None:
        progress(None)  # models ready: back to the plain spinner and timer while identifying

    photo = Path(photo)
    out_dir = Path(paths["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{photo.stem}"
    torch.cuda.reset_peak_memory_stats()
    times = process_image(photo, out_dir, run_cfg, segmenter, embedder, index, recognizer, min_confidence)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    # What this process can use: free VRAM plus what PyTorch already holds (other programs excluded)
    usable_gb = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30

    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    summary = summary_html(results, times, load_secs, models, peak_gb, peak_gb > usable_gb - 1, out_dir)
    return str(out_dir / "annotated.jpg"), summary, items_html(results, out_dir)


def db_overview(search_id):
    """Status line and per-product photo counts for the Database tab, for the selected search model."""
    try:
        entries = scan_db(paths["db_dir"])
    except FileNotFoundError:
        return (f"No database folder yet. Create `{paths['db_dir']}\\<ProductName>\\` folders with photos, "
                "then click **Rebuild index**."), []
    rows = [list(r) for r in sorted(Counter(e["item"] for e in entries).items())]
    name, index = short(search_id), load_index(search_id)
    if index is None:
        return (f"⚠️ No index for {name} yet. It is built automatically on the next Identify, "
                "or click **Rebuild index**."), rows
    added, removed = index_changes(index, entries)
    if not (added or removed):
        return f"✅ Index for {name} is up to date: {len(entries):,} photos of {len(rows)} products.", rows
    return (f"⚠️ Photos changed since the index for {name} was built ({added} added, {removed} removed). "
            "It is updated automatically on the next Identify, or click **Rebuild index**."), rows


def rebuild(search_id, progress=gr.Progress()):
    entries, vecs = build(search_id, progress)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        leave_one_out(entries, vecs, cfg["retrieval"]["top_k"])
    status, rows = db_overview(search_id)
    return status, rows, (f"Accuracy check for {short(search_id)} "
                          f"(each photo matched against all the others):\n```\n{report.getvalue()}```")


def start(method, model_ids, progress):
    """Load the chosen method's models (freeing the other method's) and show that method's page.
    The progress bar on the home page names each model while it loads."""
    kinds = METHODS[method]["kinds"]
    free_unused(kinds)
    for i, (kind, model_id) in enumerate(zip(kinds, model_ids)):
        get_model(kind, model_id, on_load=lambda m, i=i: progress((i, len(kinds)), desc=f"Loading {short(m)} ...",
                                                                  unit="models"))
    # Only the page being shown gets an update: in this Gradio version a page that receives any
    # update before it has ever been shown can never be shown again
    return (gr.update(visible=False), *(gr.update(visible=True) if key == method else gr.skip() for key in PAGES),
            method)


def start_handler(method):
    """The Start button's handler for one method. Gradio only passes the progress tracker to a named
    parameter (not one after *args), so the handler lists the page's model dropdowns one by one."""
    if len(METHODS[method]["kinds"]) == 2:
        def handler(sam_id, search_id, progress=gr.Progress()):
            return start(method, (sam_id, search_id), progress)
    else:
        def handler(sam_id, search_id, vlm_id, progress=gr.Progress()):
            return start(method, (sam_id, search_id, vlm_id), progress)
    return handler


def go_home(current):
    """Hide only the page that is showing (see start)."""
    return (gr.update(visible=True), *(gr.update(visible=False) if key == current else gr.skip() for key in PAGES))


def refresh_database(current, *db_dropdowns):
    """Database-tab status of the page that is showing; other pages are left untouched (see start)."""
    return [x for key, dd in zip(PAGES, db_dropdowns)
            for x in (db_overview(dd) if key == current else (gr.skip(), gr.skip()))]


def model_choices(kind):
    return [(short(m), m) for m in cfg["models"][kind]]


def card_html(m):
    facts = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in m["facts"])
    return (('<span class="mc-badge">Recommended</span>' if m["recommended"] else "")
            + f'<h2 class="mc-name">{m["name"]}</h2><p class="mc-steps">{m["steps"]}</p>'
            + f'<p class="mc-about">{m["about"]}</p><dl class="mc-facts">{facts}</dl>')


def build_page(method):
    """One complete page per method, every control always visible inside it. Switching methods only
    shows/hides whole pages: Gradio loses show/hide changes made to single controls of a hidden page."""
    m = METHODS[method]
    siglip = method == "siglip"
    search = "siglip" if siglip else "embed"
    search_label = "SigLIP model" if siglip else "Search model"
    with gr.Column(visible=False) as page:
        with gr.Row(equal_height=True):
            gr.HTML(f'<div class="bs-method"><span class="bs-method-name">{m["name"]}</span>'
                    f'<span class="bs-method-steps">{m["steps"]}</span></div>')
            home_btn = gr.Button("← Change method", size="sm", scale=0, min_width=150)
        with gr.Tab("Identify"):
            with gr.Row():
                # Left: what you do (photo, Identify, the few settings that matter). Right: the result.
                with gr.Column(scale=1, min_width=300):
                    photo = gr.Image(type="filepath", label="Basket photo (upload, webcam or paste)", height=300)
                    run_btn = gr.Button("Identify items", variant="primary", size="lg")
                    min_conf = gr.Slider(
                        0, 1, value=cfg["siglip"]["min_score"] if siglip else 0, step=0.05,
                        label="Minimum confidence", info="Items below this show as unknown. " + (
                            "Score = image match and name match combined." if siglip else
                            "VLM answers are usually 0.8–1.0; without the VLM it is the search score "
                            "(about 0.2–0.8)."))
                    use_vlm = None if siglip else gr.Checkbox(
                        value=True, label="Use VLM",
                        info="Slower, but checks each match and can answer 'unknown'. "
                             "Off: closest search match only.")
                    with gr.Accordion("Models", open=False):
                        sam_dd = gr.Dropdown(model_choices("sam2"), value=cfg["sam2"]["model_id"],
                                             label="Segmentation model")
                        search_dd = gr.Dropdown(model_choices(search), value=cfg[search]["model_id"],
                                                label=search_label)
                        vlm_dd = None if siglip else gr.Dropdown(model_choices("vlm"), value=cfg["vlm"]["model_id"],
                                                                 label="VLM")
                with gr.Column(scale=3):
                    summary = gr.HTML('<p class="bs-empty">Add a basket photo, then press <b>Identify items</b>.</p>')
                    with gr.Row():
                        annotated = gr.Image(label="Result", interactive=False, height=480, scale=3)
                        item_list = gr.HTML(max_height=480, scale=2, min_width=320)
        with gr.Tab("Database") as db_tab:
            db_dd = gr.Dropdown(model_choices(search), value=cfg[search]["model_id"], label=search_label,
                                info="Each search model has its own index. Same choice as on the Identify tab.")
            with gr.Row(equal_height=True):
                status = gr.Markdown(scale=4)
                rebuild_btn = gr.Button("Rebuild index", scale=0, min_width=160)
            db_items = gr.Dataframe(headers=["Product", "Photos"], interactive=False, show_search="search",
                                    max_height=420)
            report = gr.Markdown()

    # Identify and Rebuild use the GPU and swap models and indexes, so they share one queue with Start
    # (progress=gr.Progress() lets Gradio show which model is loading while you wait)
    if siglip:
        def run(photo, sam_id, search_id, min_confidence, progress=gr.Progress()):
            return identify(photo, method, False, sam_id, search_id, None, min_confidence, progress)
        inputs = [photo, sam_dd, search_dd, min_conf]
    else:
        def run(photo, use_vlm, sam_id, search_id, vlm_id, min_confidence, progress=gr.Progress()):
            return identify(photo, method, use_vlm, sam_id, search_id, vlm_id, min_confidence, progress)
        inputs = [photo, use_vlm, sam_dd, search_dd, vlm_dd, min_conf]
    run_btn.click(run, inputs, [annotated, summary, item_list], concurrency_id="gpu",
                  api_name=f"identify_{method}")
    rebuild_btn.click(rebuild, [db_dd], [status, db_items, report], concurrency_id="gpu",
                      api_name=f"rebuild_{method}")
    # The Identify and Database search dropdowns mirror each other (.input = user changes only, so no loop)
    search_dd.input(lambda m: m, search_dd, db_dd)
    db_dd.input(lambda m: m, db_dd, search_dd)
    db_dd.change(db_overview, [db_dd], [status, db_items])
    db_tab.select(db_overview, [db_dd], [status, db_items])  # Identify may have rebuilt the index meanwhile
    return page, home_btn, [dd for dd in (sam_dd, search_dd, vlm_dd) if dd is not None], db_dd, status, db_items


with gr.Blocks(title="Basket Scanner", analytics_enabled=False) as demo:
    gr.HTML('<div class="bs-header"><h1>Basket Scanner</h1>'
            '<p>Find every item in a basket photo and name it from your product database.</p></div>')
    with gr.Column(visible=True) as home:
        start_btns = {}
        with gr.Row(equal_height=True):
            for key, m in METHODS.items():
                with gr.Column(elem_classes=["method-card"] + (["recommended"] if m["recommended"] else [])):
                    gr.HTML(card_html(m))
                    start_btns[key] = gr.Button(f"Start {m['name']}",
                                                variant="primary" if m["recommended"] else "secondary")
        load_status = gr.Markdown("", min_height=90)  # Start's progress shows here: which model is loading
        gr.HTML('<p class="bs-note">Figures measured on 11 test photos with compare_methods.py and the default '
                'models. Change models under “Models” on the Identify tab.</p>')
    PAGES = {method: build_page(method) for method in METHODS}
    current = gr.State(None)  # the method page this browser is showing

    pages = [p[0] for p in PAGES.values()]
    for key, btn in start_btns.items():
        # Each Start passes its own page's model dropdowns; the progress bar shows under the cards
        btn.click(start_handler(key), PAGES[key][2], [home] + pages + [current], concurrency_id="gpu",
                  api_name=f"start_{key}", show_progress_on=[load_status]).then(
            refresh_database, [current] + [p[3] for p in PAGES.values()], [x for p in PAGES.values() for x in p[4:6]])
    for _, home_btn, *_ in PAGES.values():
        home_btn.click(go_home, current, [home] + pages)

if __name__ == "__main__":
    demo.launch(inbrowser=True, show_error=True, allowed_paths=[paths["out_dir"]], theme=THEME, css=CSS)
