# Basket Scanner (S-VLM)

Finds every item in a photo of a basket or tray and names it from a product photo database.
SAM2 cuts out each item, then one of two methods names each cut-out:

- **SigLIP** (recommended): matches the cut-out against the database photos and the product names.
  Fast and light.
- **DINOv2 + Qwen**: DINOv2 shortlists the 5 closest products, then the Qwen2.5-VL vision-language
  model picks one or answers unknown. Slower, and needs a 24 GB GPU.

## What you need

- Windows 10 or 11 (that is what it was tested on) and an NVIDIA GPU with CUDA. On an RTX 3090
  (24 GB) the SigLIP method used 5–9 GB and DINOv2 + Qwen 7B used 19–22 GB.
- Python 3.10.
- Disk space for the models: about 6 GB for the SigLIP method, 23 GB for both methods with the
  default models, 80 GB for every model choice in `config.yaml`.

## Setup

In PowerShell, in this folder:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$env:SAM2_BUILD_CUDA = '0'   # skips SAM2's optional CUDA extension, which is hard to build on Windows
.venv\Scripts\python.exe -m pip install https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py
```

`download_models.py` downloads every model listed under `models:` in `config.yaml` into `models\`
(80 GB). To download less, first delete the choices you don't need from those lists, but keep
each section's `model_id`: that is the default the app starts with. After the download everything
runs offline.

## Run the app

```powershell
.venv\Scripts\python.exe app.py
```

It opens http://127.0.0.1:7860. Pick a method on the home page; its models load (about 45 s for
Qwen, a few seconds for SigLIP). Then add a photo and press **Identify items**.

- **Models**, a closed section on the Identify tab, holds the model choices: SAM2 size, search
  model and VLM.
- **Minimum confidence**: items that score lower are shown as unknown. SigLIP starts at 0.5.
- The first time a search model is used, it builds an index of the database photos, which takes a
  few minutes. The index is rebuilt automatically when photos are added or removed. The
  **Database** tab lists the products and has a **Rebuild index** button.

## Other scripts

- `run.py photo.jpg [more.jpg | folder]`: DINOv2 + Qwen from the command line (`--no-vlm` takes
  DINOv2's closest match instead). Results go to `outputs\<photo>\`: `results.json`,
  `annotated.jpg` and the crops. Build the DINOv2 index first with `build_db.py`, or by using the
  app once.
- `build_db.py`: builds the DINOv2 index. `--eval` also checks search accuracy by matching each
  database photo against all the others.
- `compare_methods.py`: runs both methods on photos whose contents are known, and prints the items
  found, wrong names, time and GPU memory. By default it uses the 11 photos listed in
  `test\labels.yaml`; `--labels test\trays\labels.yaml` runs the 3 simulated trays.

## Product database

`db2\` holds one folder per product with its photos inside. The folder name is the product name
the app reports, and SigLIP also matches the cut-out against that name, so clear names help.
`paths: db_dir` in `config.yaml` points to the folder.

- 7 real products, with 4–7 photos each: `aa_battery`, `aaa_battery`, `charger`, `eclipse`,
  `loctite_243`, `loctite_2701`, `pen`.
- 295 stand-ins, `product_000` to `product_294`: drawn shapes labelled "BRAND 000" and so on,
  added to test speed with about 300 products. Delete them when real products are added.

## Settings

`config.yaml` holds:

- the default models: `model_id` in each section
- SAM2's settings: `sam2:`
- the filter that drops the basket, the background and small pieces: `filter:`
- SigLIP's scoring: `siglip:`
- battery size measurement: `battery_size:`, which is off.

## Where things stand

Tested 9–11 October 2026 on an RTX 3090.

| | SigLIP | DINOv2 + Qwen 7B |
|---|---|---|
| 11 test photos, 14 items, default models: named correctly | 13 | 11 |
| Wrong or extra names on those photos | 3 | 9 |
| 3 simulated trays, 24 items, larger models: named correctly | 21 | 19 |
| Time per photo, and per 2000 × 1500 tray | about 2 s and 10 s | about 5.5 s and 60 s |

On the trays, SAM2 cut out 21 of the 24 items, so 21 is the most either method could name.
With DINOv2 large instead of base, DINOv2 + Qwen also named 13 of the 14 photo items.

Known issues and next steps:

- SAM2 also cuts out things that are not products: caps, logos, pieces of background. SigLIP
  usually scores them under 0.5, so they show as unknown; Qwen often gives them product names.
- On the large tray photos SAM2 missed small or partly hidden items. Its quality limit
  (`pred_iou_thresh` 0.88) or size limit (`min_area_frac`, 0.3% of the photo) dropped them. With
  0.80 and 0.2% it found 23 of the 24, but the config has not been changed yet.
- SAM2's time grows with the photo size. Segmenting a smaller copy of the photo is probably the
  biggest speed-up; it has not been tried yet.
- The test photos are close copies of database photos, and the trays were built from them. A fair
  accuracy test needs new photos of real baskets or trays.

## Files

| File | What it does |
|---|---|
| `app.py` | The web app |
| `run.py` | Command-line runner (DINOv2 + Qwen) |
| `build_db.py` | Builds the DINOv2 index of the database |
| `compare_methods.py` | Compares the two methods on photos whose contents are known |
| `download_models.py` | Downloads the models listed in `config.yaml` |
| `svlm/segment.py` | SAM2 segmentation, filtering and cropping |
| `svlm/embed.py` | DINOv2 embeddings and the search index (FAISS) |
| `svlm/siglip.py` | SigLIP embeddings and the SigLIP method |
| `svlm/vlm.py` | The Qwen VLM decision |
| `test/` | Test photos, what each contains (`labels.yaml`), and the simulated trays |
