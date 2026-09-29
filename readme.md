# ClearSky-AI: Sentinel-2 cloud reconstruction

ClearSky-AI is a personal remote-sensing research project. Its primary demo takes cloudy Sentinel-2 optical data plus co-registered Sentinel-1 VV/VH radar and produces a 13-band Sentinel-2 estimate. The demo integrates the published SAR + CARL checkpoint by Meraner et al. (2020); it is not a ClearSky-AI-trained model. The earlier LISS-IV experiments and app remain preserved as a separate research track.

New to the project? Start with the [beginner guide](docs/BEGINNER_GUIDE.md) for setup, data, training, evaluation, and the planned custom PyTorch architecture.

This project owns the application, data preparation, and evaluation workflow—not the pretrained weights or their original architecture. Results are estimates and can hallucinate or miss surface detail hidden by clouds. See [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) for the objective, input contract, evaluation plan, and limitations.

The software is distributed under GNU GPL-3.0; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The sample dataset has its own attribution and terms described below.

## Start here: Sentinel reconstruction demo

The 1,060 paired demo samples (3,181 GeoTIFFs, about 2.44 GB) are distributed with Git LFS. On a new computer, install Git LFS before cloning so the imagery is downloaded rather than leaving pointer files:

```sh
git lfs install
git clone https://github.com/siddharthhkk/CLEARSKKYAI.git
cd CLEARSKKYAI
git lfs pull
```

Install the project dependencies and launch the app:

```sh
pip install -r requirements.txt
streamlit run app.py
```

The app accepts a 13-band cloudy Sentinel-2 GeoTIFF in `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12` order plus a co-registered two-band Sentinel-1 GeoTIFF in `VV, VH` order, with SAR values in dB. Both files must use the same pixel grid, CRS, and affine transform. The output retains the optical grid and has 13 bands. The UI supports file uploads, local paths, or the prepared paired-sample gallery.

The published SAR + CARL pretrained checkpoint is included in Git LFS at `weights/dsen2cr_sar_carl.pth`. It is the original authors' model converted to PyTorch, not a checkpoint trained by this project. The app reports when model or sample files are still Git LFS pointers and tells you to run `git lfs pull`. Its model selector also detects a separately trained local checkpoint at `weights/dsen2cr_sar_carl_trained.pth`.

Run the checkpoint/inference check and the synthetic, one-epoch training smoke test after setup:

```sh
python scripts/validate_sentinel_dsen2cr.py
python scripts/validate_dsen2cr_training.py
```

### Train the PyTorch Sentinel model

The Sentinel inference path and training entry point use PyTorch only; TensorFlow is not required. The model matches the published 13-band Sentinel-2 + 2-band Sentinel-1 residual network. `train.py` implements the paired-data loader, released optical/SAR scaling, the upstream cloud/shadow mask heuristic, and CARL loss, with validation-loss checkpointing and resumable state. The default architecture, cloud threshold, input scaling, and learning rate follow the published setup; batch size defaults to 1 to reduce memory demand. This is a maintained PyTorch training workflow, not a claim of bit-for-bit equivalence to the historical TensorFlow/Keras runtime.

Download the full SEN12MS-CR data/index from the [official dataset record](https://mediatum.ub.tum.de/1554803), then use its `datasetfilelist.csv` and extracted triplet folders:

```powershell
python train.py `
  --manifest path\to\datasetfilelist.csv `
  --data-root path\to\SEN12MS-CR `
  --output weights\dsen2cr_sar_carl_trained.pth `
  --epochs 8 --batch-size 1 --crop-size 128 --device auto
```

The listed splits `1` and `2` are used for training and validation; listed test split `3` is excluded. The run writes the best validation checkpoint, a `*_latest.pth` resume checkpoint, and an epoch history CSV. For the command above, resume with `--resume weights\dsen2cr_sar_carl_trained_latest.pth`. The project gallery contains only 96 train-split illustrations from one scene and no validation split, so it is not a training substitute and must not be used to claim generalization. Training is optional for the demo: the included published baseline is ready for inference.

### Prepare demo examples and the expanded local subset

The official [SEN12MS-CR record](https://mediatum.ub.tum.de/1554803) describes 122,218 triplets and a 272 GB full download; the reorganized [Hugging Face mirror](https://huggingface.co/datasets/Hermanni/sen12mscr) is listed at 389 GB and carries a CC BY 4.0 license. You do not need either full download for this project. The default gallery is a quick 25-case demo. The expanded option reads the first 96 patches from each of ten held-out test scenes, spanning all four seasons, plus one train-split scene: 1,056 source triplets across eleven scenes. The checked-in Git LFS gallery occupies 2.44 GB (2.27 GiB); it contains 960 split-held-out test patches and 96 train-split illustrations, plus four curated challenge cases, for 1,060 paired examples total. Patch selections are not an independent dataset or a new benchmark.

Three curated held-out examples have published cloud annotations (86.79–97.25% coverage); a fourth is labelled urban/built-up (IGBP class 13, 63.01% cloud coverage). Cloud values come from a public [SEN12MS-CR-derived annotation dataset](https://zenodo.org/records/17114706); the land-cover class comes from the original [SEN12MS labels](https://github.com/schmitt-muc/SEN12MS). The urban label is a land-cover category, not a verified city name. The mirror strips georeferencing, so these patches cannot currently be placed at a named city on a map. The selected mirror samples are included under `data/sentinel_demo/` using Git LFS; the preparation script remains available to recreate the gallery from the public source.

```sh
pip install fsspec pyarrow
python scripts/prepare_sen12mscr_demo.py
python scripts/evaluate_sen12mscr_demo.py
```

For the expanded 2–3 GB subset and 1,060 paired patches, run:

```sh
python scripts/prepare_sen12mscr_demo.py --volume
python scripts/evaluate_sen12mscr_demo.py
```

The app shows SAR, cloudy optical input, reconstruction, and paired clear reference with per-patch comparison metrics. Use the sidebar split and season filters to browse held-out scenes separately from train-split illustrations; curated high-cloud and urban/built-up cases appear first. The clear sample is co-registered, but may not be a same-time observation or perfect ground truth. `--all-test` remains a smaller 45-case option. Samples are 256×256 patches with no geospatial transform in this mirror, not map-ready products.

Each case folder contains `cloudy_s2.tif` (13 optical bands), `sar_s1_vv_vh.tif` (two SAR bands), and `clear_s2_reference.tif` (paired target); `data/sentinel_demo/manifest.csv` records the split, season, scene, and relative paths.

Labels are intentionally precise: the train-split examples demonstrate the model's training-data family, but do not prove that each exact patch appeared in the released checkpoint's training run. Test examples are from the dataset's ten published test scenes, absent from its listed training scenes; they are held-out examples from the same dataset, not an independent external dataset. Don't report a training-split score as held-out accuracy.

The expanded 1,060-case run completed on the local pretrained checkpoint. Across the 964 test-split patches, mean per-patch PSNR was 17.25 dB for cloudy input and 27.65 dB for reconstruction (+10.40 dB mean gain); 960/964 patches improved. Mean output MAE was 0.03130 and RMSE 0.04527 (reflectance normalized to 0–1); output PSNR ranged from 18.50 to 44.18 dB. The four non-improving cases were all in summer scene 119, with PSNR changes from -0.27 to -0.71 dB. These are unweighted means of patch-level metrics, not a pooled benchmark; all test patches are from the same dataset and the clear references may differ in acquisition time. The 96 train-split examples are excluded from this held-out summary.

An earlier 24-patch held-out spot-check reported mean per-patch PSNR of 17.64 dB for cloudy input and 28.39 dB for reconstruction (+10.75 dB mean gain); all 24 patches improved. Its mean output MAE was 0.02985 and RMSE 0.04267. The three heavy-cloud cases and urban/built-up case in that spot-check scored as follows:

| Held-out example | Annotated cloud | Cloudy PSNR | Output PSNR | Gain |
|---|---:|---:|---:|---:|
| Spring scene 31, p518 | 97.25% | 12.78 dB | 29.69 dB | +16.91 dB |
| Spring scene 44, p573 | 86.79% | 10.33 dB | 34.56 dB | +24.22 dB |
| Summer scene 73, p462 | 89.70% | 11.97 dB | 26.32 dB | +14.35 dB |
| Winter scene 108, p437 (urban/built-up label) | 63.01% | 32.90 dB | 33.81 dB | +0.91 dB |

These results use paired clear references that may be from a different acquisition date. The earlier spot-check and expanded run are from one dataset and are not independent external validation. Re-run `python scripts/evaluate_sen12mscr_demo.py` to reproduce the expanded scores; per-case cloudy-input and output scores are written to ignored `data/eval/sentinel_demo/metrics.csv`.

Representative patch scores from the earlier spot-check across all 13 bands:

| Split | Season / scene | MAE ↓ | RMSE ↓ | PSNR ↑ |
|---|---|---:|---:|---:|
| Train | spring / scene 101 | 0.02640 | 0.04419 | 27.09 dB |
| Test | spring / scene 106 | 0.01979 | 0.02799 | 31.06 dB |
| Test | summer / scene 73 | 0.02103 | 0.02719 | 31.31 dB |
| Test | fall / scene 139 | 0.03098 | 0.03988 | 27.98 dB |
| Test | winter / scene 63 | 0.02792 | 0.03960 | 28.05 dB |

These are reference-based scores on individual 256×256 samples, not a full benchmark result. The clear reference can differ in acquisition time; the expanded gallery's 960 split-held-out patches are all from one dataset and remain limited evidence for cross-dataset or operational generalization.

For a prepared full Sentinel product, choose the cloudy 13-band S2 file and matching VV/VH S1 file. The model uses the published preprocessing: optical DN clipped to 0–10000 and divided by 2000; VV clipped to -25–0 dB and VH to -32.5–0 dB, each scaled to 0–2. The output is clipped to model range 0–5 and rescaled to 16-bit reflectance DN. See `scripts/infer_sentinel_dsen2cr.py` for the explicit contract and tiled inference implementation.

### Preserved LISS-IV research prototype

The earlier native LISS-IV line is retained, including its data-preparation, training, evaluation scripts, and Streamlit UI. It uses a custom three-band DSen2-CR-style model and local experimental weights; it is not the published Sentinel model and does not use SAR. Run it separately with:

```sh
streamlit run app_liss4.py
```

The following sections record that established LISS-IV experimental history; they are not the current primary demo objective.

## Historical LISS-IV model and evidence

The historical LISS-IV model was a DSen2-CR-style residual CNN with:

- Input bands: `[Green, Red, NIR]` (LISS-IV `BAND2`, `BAND3`, `BAND4`)
- Expected native DN range: `0–1023`
- 256 features, 16 residual blocks, residual scale 0.1
- Optical-only inference; optional SAR support is not part of the active checkpoint

The best local LISS-IV V3R checkpoint records epoch 16 and 34.501 dB synthetic validation PSNR. The generated V3R data has 512 patches from one clear source scene: 452 training patches, 51 spatially held-out validation patches, and 9 boundary patches discarded from the split.

An expanded two-scene candidate was fine-tuned from that checkpoint through epoch 20. On the combined 315-patch spatial validation set, the original checkpoint scores 27.801 dB and the candidate 32.863 dB. The gain comes mainly from the added Bhoonidhi scene: per-source PSNR changes from 34.501 to 32.302 dB on the original validation patches, and from 27.187 to 32.980 dB on Bhoonidhi patches. The candidate therefore adapts to the added source but trades off some original-source performance; it remains an experiment, not the default model.

Fresh tiled inference with that checkpoint was checked against the local Guwahati temporal pair:

| Input/output | MAE | RMSE | PSNR | SSIM | SAM |
|---|---:|---:|---:|---:|---:|
| Cloudy input | 0.016918 | 0.040084 | 27.941 dB | 0.94017 | 4.6183° |
| V3R single-scene baseline | 0.009091 | 0.019073 | 34.392 dB | 0.96839 | 3.5060° |
| V3R multi-scene candidate | 0.009546 | 0.022117 | 33.105 dB | 0.95867 | 3.4794° |

On this pair the single-scene baseline is better on MAE, RMSE, PSNR, and SSIM; the multi-scene candidate has a slightly lower SAM. The baseline's PSNR gain over the cloudy input is +6.451 dB, versus +5.164 dB for the candidate. This is a temporal-reference result, not simultaneous cloud-free ground truth: illumination, registration, land-surface, and acquisition differences contribute to all metrics. It does not establish performance on arbitrary scenes or cloud types.

A separate Bhoonidhi stress test (01-Jun-2020 cloudy vs 01-Nov-2023 clear) scored 8.585 dB before and 7.816 dB after V3R. The acquisitions are more than three years apart, so this is not a clean benchmark; it does show that V3R can be too conservative under a difficult cloud regime. Bhoonidhi is useful for later testing, but is not needed to run the project.

## LISS-IV reproduction and checks (legacy)

Install the Python dependencies from the project root:

```sh
pip install -r requirements.txt
```

### Streamlit demo

Launch the local interface from the repository root:

```sh
streamlit run app_liss4.py
```

The app uses this project's V3R checkpoint at `weights/liss4_dsen2cr_synthetic_v3r.pth` (download it with `git lfs pull`). Then provide either a stacked three-band GeoTIFF or three separate co-registered TIFFs in `[Green, Red, NIR]` (`BAND2`, `BAND3`, `BAND4`) order. You can upload TIFFs or enter local file paths; local paths are recommended for large scenes. The app shows a downsampled false-color input/output preview and lets you download the georeferenced reconstruction. Satellite imagery and training data remain local and are not included in the repository. This preserved LISS-IV UI is a demo wrapper around the tiled inference script documented below; results are experimental estimates, not guaranteed cloud-free ground truth.

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

For products delivered as three separate, co-registered single-band files, pass the files in Green, Red, NIR order (typically `BAND2`, `BAND3`, `BAND4`) instead of creating a stacked intermediate:

```sh
python scripts/infer_liss4_real_cloudy_only_overlap.py \
  --checkpoint weights/liss4_dsen2cr_synthetic_v3r.pth \
  --cloudy-bands path/to/BAND2.tif path/to/BAND3.tif path/to/BAND4.tif \
  --output data/eval/reconstructed.tif \
  --tile 256 --overlap 64 --device auto
```

The three input grids must match. Larger overlap can reduce tile seams, at a substantial runtime cost; inspect a preview on unfamiliar, cloud-heavy scenes. Without a clear reference, this remains a qualitative reconstruction, not an accuracy evaluation.

The project-trained LISS-IV V3R checkpoint is included through Git LFS; run `git lfs pull` after cloning. The original satellite imagery and generated training data remain excluded from Git. You can also train a replacement checkpoint locally as described below.

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

The extractor refuses to overwrite existing generated files unless `--overwrite` is supplied. V3R currently requires 256×256 source patches; patches are skipped if any of the three bands contains nodata. The above 1024×1024 example produces 49 patches and a spatial split of 35 train / 7 validation patches, with boundary-crossing patches discarded. That small single-scene set is useful for exercising the pipeline, not evidence of broad generalization.

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

## Expanded two-scene training experiment

The local expanded experiment adds a second clear-source scene while retaining the original 512-sample V3R dataset. It does not use `synthetic_pretrain_v3r_from_guwahati`: those patches come from the Guwahati evaluation area and remain excluded to avoid leakage.

The added source is the local Bhoonidhi Resourcesat-2 LISS-IV scene `R2F01NOV2023065046011000054SSANSTUC00GTDB` (01-Nov-2023, path/row 110/054). Its three separate 18,343×16,229 GeoTIFFs have matching dimensions, CRS, and affine transform and are read as `[BAND2, BAND3, BAND4]` = `[Green, Red, NIR]`. The extractor streams windows directly from those files; it does not create a full-scene stack.

Source patches are 256×256 with a 512-pixel stride. An optional strict bright-neutral screen excludes patches with more than 0.5% proxy-positive pixels and stores the proxy fraction in the manifests. This is a heuristic screen, not a cloud mask or a claim that retained patches are cloud-free. The observed run kept 1,214 of 1,221 candidate patches; the spatial split contains 950 training and 264 validation patches. The original source contributes 452 training and 51 validation patches, so the combined experiment uses 1,402 training and 315 validation samples. Validation is spatially held out within the two training scenes; it is not scene-level or real-cloud validation.

The source `BAND_META.txt` identifies the OTS product and acquisition. Its `ACC_REP.txt` has a different numeric `ProductID`; the same kind of difference appears in other locally inspected Bhoonidhi bundles. The experiment was keyed to the matching folder/OTS product ID, date, path/row, sensor, and co-registered raster grids. The accuracy-report ID crosswalk was not independently verified, so keep this as an experimental data-provenance caveat.

Recreate the local dataset when the source product is present:

```sh
python scripts/extract_liss4_clear_patches.py \
  --clear-bands \
  data/raw/bhoonidhi_liss4/R2F01NOV2023065046011000054SSANSTUC00GTDB/BAND2.tif \
  data/raw/bhoonidhi_liss4/R2F01NOV2023065046011000054SSANSTUC00GTDB/BAND3.tif \
  data/raw/bhoonidhi_liss4/R2F01NOV2023065046011000054SSANSTUC00GTDB/BAND4.tif \
  --output data/synthetic_pretrain_clear_bhoonidhi_11054_2023-11-01 \
  --patch-size 256 --stride 512 \
  --max-strict-cloud-proxy-fraction 0.005

python scripts/make_liss4_synthetic_pretrain_v3r_from_npz.py \
  --manifest data/synthetic_pretrain_clear_bhoonidhi_11054_2023-11-01/manifest.csv \
  --output data/synthetic_pretrain_v3r_bhoonidhi_11054_2023-11-01

python scripts/split_liss4_synthetic_manifest.py \
  --manifest data/synthetic_pretrain_v3r_bhoonidhi_11054_2023-11-01/manifest.csv \
  --scene-width 18343 --val-fraction 0.20
```

Train or continue the expanded experiment with repeatable train and validation manifest options. The local run continued the epoch-16 baseline to epoch 20, using a fresh optimizer at `5e-5`; the trainer resets best-checkpoint selection when the validation source changes and records dataset/configuration provenance in the checkpoint:

```sh
python scripts/train_liss4_synthetic.py \
  --train-manifest data/synthetic_pretrain_v3r/train_manifest.csv \
  --train-manifest data/synthetic_pretrain_v3r_bhoonidhi_11054_2023-11-01/train_manifest.csv \
  --val-manifest data/synthetic_pretrain_v3r/val_manifest.csv \
  --val-manifest data/synthetic_pretrain_v3r_bhoonidhi_11054_2023-11-01/val_manifest.csv \
  --epochs 20 --batch-size 1 --grad-accum 4 --lr 5e-5 \
  --features 256 --blocks 16 --lambda-cloud 2.0 \
  --resume weights/liss4_dsen2cr_synthetic_v3r.pth \
  --output weights/liss4_dsen2cr_synthetic_v3r_multiscene.pth \
  --latest-output weights/liss4_dsen2cr_synthetic_v3r_multiscene_latest.pth
```

Compare both checkpoints on the same combined synthetic validation set with:

```sh
python scripts/evaluate_liss4_synthetic_checkpoint.py \
  --checkpoint weights/liss4_dsen2cr_synthetic_v3r.pth \
  --manifest data/synthetic_pretrain_v3r/val_manifest.csv \
  --manifest data/synthetic_pretrain_v3r_bhoonidhi_11054_2023-11-01/val_manifest.csv

python scripts/evaluate_liss4_synthetic_checkpoint.py \
  --checkpoint weights/liss4_dsen2cr_synthetic_v3r_multiscene.pth \
  --manifest data/synthetic_pretrain_v3r/val_manifest.csv \
  --manifest data/synthetic_pretrain_v3r_bhoonidhi_11054_2023-11-01/val_manifest.csv
```

The raw Bhoonidhi product, generated NPZs, and checkpoints remain local and are not included in Git. The Guwahati temporal pair is still the only real-scene evaluation in this project; it is nearby and acquired on different dates, so its metrics are not same-date ground truth or evidence of broad geographic generalization.

## Optional Guwahati evaluation

When the local cloudy/clear pair and checkpoints are available, evaluate the output against the temporal clear reference:

```sh
python scripts/evaluate_liss4_real_multimetric.py \
  --cloudy data/raw/cloudy/guwahati_cloudy_test.tif \
  --clear data/raw/clear/guwahati_clear.tif \
  --v2 data/eval/guwahati_liss4_v3r_baseline_recheck.tif \
  --v3r data/eval/guwahati_liss4_v3r_multiscene.tif \
  --baseline-label V3R-single-scene-baseline \
  --candidate-label V3R-multiscene-epoch20
```

The evaluator reports MAE, RMSE, PSNR, per-band SSIM, and spectral angle mapper (SAM). Metrics assume the two images share a grid and are normalized by the fixed 1023 DN ceiling.

## Optional Bhoonidhi workflow

Bhoonidhi products are kept local and ignored by Git. This checkout has a small locally downloaded Resourcesat-2 LISS-IV set; the existing tools can scan products, compare cloudiness heuristics, inspect overlaps, create previews, and run the V3R stress test:

```sh
python scripts/scan_bhoonidhi_liss4.py
python scripts/find_bhoonidhi_liss4_pairs.py
python scripts/compare_bhoonidhi_11054_cloudiness.py
```

Pair-specific commands are available in each script's `--help`. Cloudiness comparisons are heuristics for candidate screening, not native cloud masks or accuracy claims.

## Data and model contract

- The primary Sentinel demo requires 13 co-registered Sentinel-2 bands and two co-registered Sentinel-1 bands; see [Start here](#start-here-sentinel-reconstruction-demo).
- The legacy LISS-IV interface accepts a stacked three-band GeoTIFF or three co-registered single-band files in `[Green, Red, NIR]` order.
- Native LISS-IV processing uses one fixed `dn_max=1023`; per-image min/max normalization is intentionally avoided.
- Inspect unfamiliar products with `scripts/inspect_liss4.py` before inference or training.
- Train/clear pairs must be co-registered on an identical grid. The paired dataset loader rejects grid mismatches.
- Real temporal pairs are references, not pixel-perfect ground truth.
- Local `data/` is excluded from Git; only the demo Sentinel samples and the documented model checkpoints are versioned through Git LFS. Other local experiment checkpoints remain ignored.

## Repository map

- `app.py`: ClearSky-AI Sentinel-1/2 Streamlit demo
- `app_liss4.py`: preserved LISS-IV prototype interface
- `PROBLEM_STATEMENT.md`: current problem definition and evaluation scope
- `src/dsen2cr.py`: PyTorch implementation of the published Sentinel DSen2-CR architecture
- `scripts/infer_sentinel_dsen2cr.py`: aligned Sentinel GeoTIFF inference
- `scripts/prepare_sen12mscr_demo.py`: selected public SEN12MS-CR gallery preparation
- `scripts/evaluate_sen12mscr_demo.py`: per-case metrics on prepared reference pairs
- `scripts/validate_sentinel_dsen2cr.py`: checkpoint and integration smoke check
- `src/liss4.py`, `src/liss4_dsen2cr.py`: legacy LISS-IV DN normalization and custom model
- `src/liss4_dataset.py`, `src/liss4_patch_dataset.py`: paired and patch data loaders
- `scripts/extract_liss4_clear_patches.py`: clear GeoTIFF to NPZ source patches
- `scripts/make_liss4_synthetic_pretrain_v3r_from_npz.py`: V3R synthetic corruption
- `scripts/train_liss4_synthetic.py`: training and checkpoint writing
- `scripts/evaluate_liss4_synthetic_checkpoint.py`: same-split synthetic checkpoint comparison
- `scripts/infer_liss4_real_cloudy_only_overlap.py`: tiled GeoTIFF inference
- `scripts/evaluate_liss4_real_multimetric.py`: quantitative temporal-reference metrics
- `scripts/validate_liss4_*.py`: setup, data, training, and inference checks

The published Sentinel-2 checkpoint is included in Git LFS, alongside the separately documented project-trained LISS-IV checkpoint. The original authors' repository documents the Sentinel model and preprocessing lineage; SEN12MS-CR publishes the paired S1/cloudy-S2/clear-S2 data used for that task. Cite the [original model project](https://github.com/ameraner/dsen2-cr) and [SEN12MS-CR record](https://mediatum.ub.tum.de/1554803) when presenting the Sentinel-2 project.

## Known limitations

- The published model reconstructs rather than observes hidden surface detail; the estimate can be inaccurate, especially under thick cloud, haze, snow, or domain shift.
- The prepared gallery comes from one dataset. Its test scenes are held-out examples, not independent external validation; the train-split example is not a held-out score.
- User-supplied scenes must already match the model's optical order and reflectance scaling, SAR VV/VH order and dB scaling, and exact pixel grid.
- The complete SEN12MS-CR archive is 272 GB; the optional local gallery is only a small subset and its mirror omits georeferencing.
- No claim is made that these published weights are locally retrained, optimal for Indian imagery, or suitable for LISS-IV.
- Large Sentinel rasters use overlapping tiles but currently allocate output-sized blending buffers; begin with gallery patches or modest test rasters.

### Historical LISS-IV limitations

- Synthetic training uses patches from two source scenes, but validation is spatially held out within those same scenes; scene-level generalization remains untested.
- The multi-scene candidate improves combined synthetic validation but underperforms the original baseline on the held-out Guwahati temporal pair.
- The cloud screen is heuristic only; reliable native cloud masks and simultaneous cloudy/clear LISS-IV pairs remain scarce.
- Native paired cloudy/clear acquisitions and reliable cloud masks are scarce.
- The historical LISS-IV checkpoint does not use SAR.
- LISS-IV large-scene runtime and row-buffer memory depend on scene width, overlap, and tile size.
