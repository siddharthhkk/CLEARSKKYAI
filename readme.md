# ClearSky-AI: LISS-IV cloud removal

ClearSky-AI is a personal research prototype for reconstructing cloud-affected native LISS-IV imagery. It is motivated by the ISRO hackathon problem, but is being developed as a practical personal tool rather than a strict competition submission.

The repository keeps the established optical-only DSen2-CR-style V3R model as its baseline. Bhoonidhi is optional: the inference, data-preparation, training, and smoke-test workflows do not require Bhoonidhi products.

## Current model and evidence

The active model is a DSen2-CR-style residual CNN with:

- Input bands: `[Green, Red, NIR]` (LISS-IV `BAND2`, `BAND3`, `BAND4`)
- Expected native DN range: `0–1023`
- 256 features, 16 residual blocks, residual scale 0.1
- Optical-only inference; optional SAR support is not part of the active checkpoint

The best local V3R checkpoint records epoch 16 and 34.501 dB synthetic validation PSNR. The generated V3R data has 512 patches from one clear source scene: 452 training patches, 51 spatially held-out validation patches, and 9 boundary patches discarded from the split.

Fresh tiled inference with that checkpoint was checked against the local Guwahati temporal pair:

| Input/output | MAE | RMSE | PSNR | SSIM | SAM |
|---|---:|---:|---:|---:|---:|
| Cloudy input | 0.016918 | 0.040084 | 27.941 dB | 0.94017 | 4.6183° |
| V3R output | 0.009091 | 0.019073 | 34.392 dB | 0.96839 | 3.5060° |

The observed PSNR difference is +6.451 dB for this pair. This is a temporal-reference result, not simultaneous cloud-free ground truth: illumination, registration, land-surface, and acquisition differences contribute to all metrics. It does not establish performance on arbitrary scenes or cloud types.

A separate Bhoonidhi stress test (01-Jun-2020 cloudy vs 01-Nov-2023 clear) scored 8.585 dB before and 7.816 dB after V3R. The acquisitions are more than three years apart, so this is not a clean benchmark; it does show that V3R can be too conservative under a difficult cloud regime. Bhoonidhi is useful for later testing, but is not needed to run the project.

## Quick start

Install the Python dependencies from the project root:

```sh
pip install -r requirements.txt
```

Run the self-contained checks. These create temporary fixtures and do not need Bhoonidhi, a downloaded dataset, or the trained checkpoint:

```sh
python scripts/validate_liss4_setup.py
python scripts/validate_liss4_data_prep.py
python scripts/validate_liss4_training.py
python scripts/validate_liss4_inference.py
```

For a real inference run, provide a three-band GeoTIFF in `[Green, Red, NIR]` order and a local model checkpoint:

```sh
python scripts/inspect_liss4.py path/to/cloudy_liss4.tif
python scripts/infer_liss4_real_cloudy_only_overlap.py \
  --checkpoint weights/liss4_dsen2cr_synthetic_v3r.pth \
  --cloudy path/to/cloudy_liss4.tif \
  --output data/eval/reconstructed.tif \
  --tile 256 --overlap 64 --device auto
```

The inference command accepts `auto`, `cpu`, or `cuda`. It writes a georeferenced `uint16` GeoTIFF, keeps the source CRS/transform and band labels, and processes overlapping tiles with a rolling tile-row buffer rather than loading the full scene into float arrays. Memory still grows with the input image width and tile size. No clear reference is used during inference.

The checkpoint and imagery are intentionally excluded from Git. On a fresh clone, put a checkpoint at the path above or train one locally as described below; do not expect GitHub to contain the ignored local weights or data.

## Prepare training data from a local clear GeoTIFF

The preparation pipeline can start from any suitable stacked three-band clear LISS-IV GeoTIFF; no Bhoonidhi data or existing NPZ cache is required. For example, this project has a local 1024×1024 Guwahati clear image:

```sh
python scripts/extract_liss4_clear_patches.py \
  data/raw/clear/guwahati_clear.tif \
  --output data/synthetic_pretrain_clear \
  --patch-size 256 --stride 128

python scripts/make_liss4_synthetic_pretrain_v3r_from_npz.py \
  --manifest data/synthetic_pretrain_clear/manifest.csv \
  --output data/synthetic_pretrain_v3r_from_guwahati

python scripts/split_liss4_synthetic_manifest.py \
  --manifest data/synthetic_pretrain_v3r_from_guwahati/manifest.csv \
  --scene-width 1024 --val-fraction 0.25

python scripts/validate_liss4_synthetic_pretrain.py \
  --manifest data/synthetic_pretrain_v3r_from_guwahati/manifest.csv \
  --samples-dir data/synthetic_pretrain_v3r_from_guwahati
```

The extractor refuses to overwrite existing generated files unless `--overwrite` is supplied. The above 1024×1024 example produces 49 patches and a spatial split of 35 train / 7 validation patches, with boundary-crossing patches discarded. That small single-scene set is useful for exercising the pipeline, not evidence of broad generalization.

Train from those split manifests with:

```sh
python scripts/train_liss4_synthetic.py \
  --manifest data/synthetic_pretrain_v3r_from_guwahati/manifest.csv \
  --train-manifest data/synthetic_pretrain_v3r_from_guwahati/train_manifest.csv \
  --val-manifest data/synthetic_pretrain_v3r_from_guwahati/val_manifest.csv \
  --output weights/liss4_dsen2cr_synthetic_from_clear.pth
```

Training uses the GPU when available. The trainer's no-extra-arguments default points to `data/synthetic_pretrain_v3r/manifest.csv` and automatically uses sibling `train_manifest.csv` and `val_manifest.csv`; specify explicit manifests when your data uses another directory. The best checkpoint and a resumable `*_latest.pth` are written under `weights/`.

The local 512-sample V3R experiment was trained from a substantially larger source-patch set than the small Guwahati bootstrap example. Synthetic-cloud validation alone is not a real-cloud benchmark.

## Optional Guwahati evaluation

When the local cloudy/clear pair and checkpoints are available, evaluate the output against the temporal clear reference:

```sh
python scripts/evaluate_liss4_real_multimetric.py \
  --cloudy data/raw/cloudy/guwahati_cloudy_test.tif \
  --clear data/raw/clear/guwahati_clear.tif \
  --v2 data/eval/guwahati_liss4_dsen2cr_v2.tif \
  --v3r data/eval/guwahati_liss4_dsen2cr_v3r_streamed_final.tif
```

The evaluator reports MAE, RMSE, PSNR, per-band SSIM, and spectral angle mapper (SAM). Metrics assume the two images share a grid and are normalized by the fixed 1023 DN ceiling.

## Optional Bhoonidhi workflow

Bhoonidhi products are kept local and ignored by Git. If downloaded later, the existing tools can scan products, compare cloudiness heuristics, inspect overlaps, create previews, and run the V3R stress test:

```sh
python scripts/scan_bhoonidhi_liss4.py
python scripts/find_bhoonidhi_liss4_pairs.py
python scripts/compare_bhoonidhi_11054_cloudiness.py
```

Pair-specific commands are available in each script's `--help`. Cloudiness comparisons are heuristics for candidate screening, not native cloud masks or accuracy claims.

## Data and model contract

- Supply a stacked three-band GeoTIFF in `[Green, Red, NIR]` order.
- Native LISS-IV processing uses one fixed `dn_max=1023`; per-image min/max normalization is intentionally avoided.
- Inspect unfamiliar products with `scripts/inspect_liss4.py` before inference or training.
- Train/clear pairs must be co-registered on an identical grid. The paired dataset loader rejects grid mismatches.
- Real temporal pairs are references, not pixel-perfect ground truth.
- Local `data/` and model checkpoint files under `weights/` are excluded from Git; only code, configs, and documentation are versioned.

## Repository map

- `src/liss4.py`, `src/liss4_dsen2cr.py`: DN normalization and active model
- `src/liss4_dataset.py`, `src/liss4_patch_dataset.py`: paired and patch data loaders
- `scripts/extract_liss4_clear_patches.py`: clear GeoTIFF to NPZ source patches
- `scripts/make_liss4_synthetic_pretrain_v3r_from_npz.py`: V3R synthetic corruption
- `scripts/train_liss4_synthetic.py`: training and checkpoint writing
- `scripts/infer_liss4_real_cloudy_only_overlap.py`: tiled GeoTIFF inference
- `scripts/evaluate_liss4_real_multimetric.py`: quantitative temporal-reference metrics
- `scripts/validate_liss4_*.py`: setup, data, training, and inference checks

The repository also retains `src/dsen2cr.py` and its converter as an optional reference to the published DSen2-CR implementation. The upstream checkpoint is not included.

## Known limitations

- Current synthetic training evidence comes from one source scene; more diverse clear LISS-IV sources are needed.
- Cloud realism and transfer to unseen regions/cloud regimes remain unproven.
- Native paired cloudy/clear acquisitions and reliable cloud masks are scarce.
- The active checkpoint does not use SAR.
- Large scenes are processed without full-scene float buffers, but runtime and row-buffer memory still depend on scene width, overlap, and tile size.
