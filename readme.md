# ClearSky-AI

ClearSky-AI is a research prototype for generative cloud removal and reconstruction of native LISS-IV satellite imagery, developed for the 2026 Bharatiya Antariksh Hackathon problem.

The repository is now focused on the native LISS-IV pipeline. The earlier Sentinel-2/U-Net prototype and superseded experiment scripts have been removed from the active tree.

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

V3R was created after measuring the corruption distribution of a real Guwahati LISS-IV cloudy/clear pair.

Earlier synthetic V2 corruption was much too strong. V3R keeps the multi-scale cloud geometry but reduces cloud radiance, opacity, shadow strength, and noise so the synthetic corruption magnitude is closer to the observed real-scene distribution.

Generator:

    scripts/make_liss4_synthetic_pretrain_v3r_from_npz.py

Training:

    scripts/train_liss4_synthetic.py

V3R synthetic validation reached 34.50 dB PSNR.

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

The local Bhoonidhi workflow found a strong overlapping Path/Row 110/54 pair:

    01-Jun-2020  RS2
    01-Nov-2023  RS2

The two scenes cover about 95% of each other's footprint. The June acquisition is heavily clouded while the November acquisition is visually much clearer.

V3R on the common footprint produced:

    Cloudy: MAE 0.28845 | RMSE 0.37218 | PSNR 8.585 dB | SSIM 0.38228 | SAM 24.8345°
    V3R:    MAE 0.29977 | RMSE 0.40665 | PSNR 7.816 dB | SSIM 0.38970 | SAM 25.1617°

This is recorded as a stress test rather than a clean benchmark because the acquisitions are more than three years apart and have substantial radiometric and surface differences.

The next validation target is a cloudy 110/54 acquisition much closer in time to the 01-Nov-2023 clear observation.

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

Compare the current 110/54 acquisitions:

    python scripts/compare_bhoonidhi_11054_cloudiness.py

Preview a pair:

    python scripts/make_bhoonidhi_pair_preview.py <cloudy_scene_id> <clear_scene_id>

Run V3R on a real pair:

    python scripts/evaluate_bhoonidhi_pair_v3r.py <cloudy_scene_id> <clear_scene_id>

Create the visual diagnostic:

    python scripts/make_bhoonidhi_v3r_diagnostic.py <cloudy_scene_id> <clear_scene_id>

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

## Important validation lesson

A public Chennai cloudy sample was previously tested and found to contain a strong periodic 128-pixel checkerboard artifact already present in the source TIFF. It is therefore not used as validation evidence.

The project now prioritizes real products with verified metadata, complete bands, and measurable geographic overlap.

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

## Next step

Find a near-date cloudy 110/54 Bhoonidhi acquisition around the clear 01-Nov-2023 observation, then use it as the next independent temporal validation case.

The longer-term target is a larger multi-AOI real-cloud LISS-IV benchmark with native cloud masks and closely matched temporal observations.
