# ClearSky-AI

ClearSky-AI is a research prototype for generative cloud removal and reconstruction of native LISS-IV satellite imagery, developed for the 2026 Bharatiya Antariksh Hackathon problem.

The repository is now focused on the native LISS-IV pipeline. Superseded Sentinel-2/U-Net experiments and exploratory gating/V2/V3 variants have been removed from the active tree.

## Active model

The current primary model is a DSen2-CR-style residual network adapted to LISS-IV.

    Input:  [G, R, NIR]
    DN:    0–1023
    Width: 256 features
    Blocks: 16 residual blocks
    Residual scale: 0.1
    Output: cloudy + predicted residual

The active experiments use optical-only LISS-IV. The model implementation keeps optional 2-channel SAR support for later work.

Core files:

    src/liss4.py
    src/liss4_dsen2cr.py
    src/liss4_dataset.py
    src/liss4_patch_dataset.py

## V3R synthetic pretraining

V3R was created after measuring the corruption distribution of a real Guwahati LISS-IV cloudy/clear temporal pair.

Earlier synthetic V2 corruption was much too strong. V3R keeps the multi-scale cloud geometry but reduces cloud radiance, opacity, shadow strength, and noise so the synthetic corruption magnitude is closer to the observed real-scene distribution.

Generator:

    scripts/make_liss4_synthetic_pretrain_v3r_from_npz.py

Training:

    scripts/train_liss4_synthetic.py

V3R synthetic validation reached 34.50 dB PSNR.

The local V3R source patches live in:

    data/synthetic_pretrain_prod/

This directory is local/ignored because it contains generated NPZ data, but it is still the source dataset required to reproduce the V3R synthetic set.

## Real validation

### Guwahati

The public Guwahati temporal pair is currently the strongest quantitative real-scene result.

    Cloudy input: 27.941 dB PSNR
    V3R output:   33.905 dB PSNR
    Gain:         +5.964 dB

Additional V3R metrics:

    MAE:  0.00943
    RMSE: 0.02017
    SSIM: 0.96237
    SAM:  3.56°

This is a temporal reference rather than simultaneous ground truth, so the metrics also contain illumination, registration, seasonal, and land-surface differences.

### Bhoonidhi / Resourcesat

A Bhoonidhi search produced several complete LISS-IV acquisitions for Path/Row 110/54.

The current complete scenes include:

    01-Jun-2020  RS2
    08-Jun-2020  RS2A
    12-Aug-2020  RS2
    01-Nov-2023  RS2

The 01-Jun-2020 acquisition is visibly/cloud-statistically cloud-heavy. The 08-Jun-2020 and 12-Aug-2020 acquisitions are even more cloud-heavy according to the current heuristic cloudiness diagnostic, while 01-Nov-2023 is very clear.

A 01-Jun-2020 → 01-Nov-2023 temporal pair has about 95% footprint overlap. V3R on the common footprint produced:

    Cloudy: MAE 0.28845 | RMSE 0.37218 | PSNR 8.585 dB | SSIM 0.38228 | SAM 24.8345°
    V3R:    MAE 0.29977 | RMSE 0.40665 | PSNR 7.816 dB | SSIM 0.38970 | SAM 25.1617°

This pair is recorded as a stress test rather than a clean benchmark. The acquisitions are more than three years apart and have large radiometric/surface differences. The result shows that the current V3R model is too conservative for this severe real-cloud case.

The next target is a cloudy 110/54 acquisition close in time to the clear 01-Nov-2023 acquisition.

## Bhoonidhi workflow

Raw Bhoonidhi products stay local and are ignored by Git.

Expected structure:

    data/raw/bhoonidhi_liss4/
        product-folder/
            BAND2.tif
            BAND3.tif
            BAND4.tif

Scan products:

    python scripts/scan_bhoonidhi_liss4.py

Find overlapping footprints:

    python scripts/find_bhoonidhi_liss4_pairs.py

Compare all current 110/54 scenes:

    python scripts/compare_bhoonidhi_11054_cloudiness.py

Preview a pair:

    python scripts/make_bhoonidhi_pair_preview.py <cloudy_scene_id> <clear_scene_id>

Run V3R on a real pair:

    python scripts/evaluate_bhoonidhi_pair_v3r.py <cloudy_scene_id> <clear_scene_id>

Create the visual diagnostic:

    python scripts/make_bhoonidhi_v3r_diagnostic.py <cloudy_scene_id> <clear_scene_id>

The cloudiness scripts are heuristic diagnostics only. They are useful for screening candidates, not for claiming a ground-truth cloud percentage.

## Synthetic-data workflow

The current workflow is:

    native LISS-IV clear imagery
            ↓
    controlled synthetic clouds
            ↓
    V3R pretraining
            ↓
    real cloudy/clear validation

Validate a generated dataset with:

    python scripts/validate_liss4_synthetic_pretrain.py

Train with the active LISS-IV trainer:

    python scripts/train_liss4_synthetic.py

## LISS-IV data contract

The project uses the fixed band order:

    BAND2 → Green
    BAND3 → Red
    BAND4 → NIR

Native products are treated as 10-bit data with a fixed ceiling of 1023.

Normalization is:

    normalized = clip(DN, 0, 1023) / 1023

Per-image min/max normalization is intentionally avoided.

Inspect a new product before using it:

    python scripts/inspect_liss4.py path/to/product.tif

## Important validation lessons

### Do not trust a public sample blindly

A public Chennai cloudy sample was previously tested and found to contain a strong periodic 128-pixel checkerboard artifact already present in the source TIFF. It is therefore not used as validation evidence.

### Temporal pairs are not ground truth

A cloudy acquisition and a clear acquisition from different dates can contain:

    illumination differences
    seasonal changes
    land-cover changes
    geometric/registration differences
    radiometric differences

Metrics from such pairs should therefore be reported as temporal-reference results, not as exact pixel-level ground truth.

### Dataset domain matters

V2 synthetic clouds were substantially stronger than the measured real-scene corruption distribution. V3R reduced the synthetic corruption amplitude and produced a large improvement on the Guwahati real pair, but the difficult Bhoonidhi stress test shows that the current model does not yet generalize to every real cloud regime.

## External DSen2-CR reference

src/dsen2cr.py contains a PyTorch implementation of the published DSen2-CR residual architecture.

The original checkpoint is not redistributed in this repository. Use scripts/convert_dsen2cr_weights.py to convert the upstream checkpoint locally when needed.

Upstream project:

    https://github.com/ameraner/dsen2-cr

## Repository structure

    CLEARSKKYAI/
    ├── configs/
    │   └── liss4_manifest.csv.example
    ├── data/                         local data; ignored
    ├── previews/                     selected visual experiment outputs
    ├── scripts/                      active LISS-IV tools
    ├── src/
    │   ├── dsen2cr.py
    │   ├── liss4.py
    │   ├── liss4_dataset.py
    │   ├── liss4_dsen2cr.py
    │   └── liss4_patch_dataset.py
    ├── weights/
    │   └── README.md
    ├── .gitignore
    ├── requirements.txt
    └── readme.md

## Requirements

Install:

    pip install -r requirements.txt

The current LISS-IV pipeline needs PyTorch, NumPy, Rasterio and Matplotlib. h5py is retained for optional conversion of the published DSen2-CR checkpoint.

## Known limitations

- Native paired LISS-IV cloudy/clear data are scarce.
- V3R training still uses synthetic cloud corruption.
- Current real references are temporal rather than simultaneous.
- Full-scene inference must be tiled and can take substantial time.
- The active V3R experiments do not yet use SAR.
- Heuristic cloudiness screening is not a substitute for a native cloud mask.

## Next step

Find a cloudy 110/54 Bhoonidhi acquisition near the clear 01-Nov-2023 observation, then evaluate V3R on that closer temporal pair.

The longer-term target is a larger multi-AOI real-cloud LISS-IV benchmark with native cloud masks and closely matched temporal observations.
