# LISS-IV DSen2-CR Adaptation — Phase 1

## Goal

Adapt the DSen2-CR residual reconstruction idea from Sentinel-1/Sentinel-2 to LISS-IV cloud removal without pretending the Sentinel-2 checkpoint is a LISS-IV model.

## Phase 1 data contract

LISS-IV multispectral imagery is treated as three optical channels:

- Green
- Red
- NIR

The model uses a fixed optical channel order:

    [G, R, NIR]

A fixed DN ceiling is applied to both cloudy and clear images in a pair. The default is 1023 for a 10-bit product, but this must be verified against the actual supplied product.

The project does not silently apply per-image min/max normalization because that would change the spectral scale from scene to scene.

## Pair manifest

Training pairs are defined explicitly in configs/liss4_manifest.csv, using the example file as a template.

Required:

    cloudy,clear,aoi,split

Optional:

    mask,sar_vv,sar_vh

The split is assigned at the AOI/scene level. Individual patches from one AOI must not be spread across train/validation/test.

## Model

src/liss4_dsen2cr.py implements a DSen2-CR-style residual network:

    cloudy optical (+ optional SAR)
                 |
          input convolution
                 |
          16 residual blocks
                 |
          residual prediction
                 |
        cloudy optical + residual

The default model uses 256 features and 16 residual blocks to stay close to the published DSen2-CR capacity. Phase-1 tests use a much smaller model only to verify tensor flow.

## Phase gates

### Gate 1 — file format

Run:

    python scripts/inspect_liss4.py path/to/file.tif

Do not proceed until:

- the file is confirmed multispectral LISS-IV;
- the three bands are identified and ordered;
- DN/radiometric scaling is understood;
- CRS, resolution, nodata and dimensions are recorded.

### Gate 2 — pair integrity

The manifest loader refuses mismatched cloudy/clear grids. Reprojection/co-registration will be added explicitly if the real data require it.

### Gate 3 — model wiring

Run:

    python scripts/validate_liss4_setup.py

This verifies normalization, both optical-only and SAR-fusion tensor paths, and GeoTIFF I/O.

### Gate 4 — training

Training is intentionally not started by Phase 1. We first need at least one real LISS-IV cloudy/clear pair and a verified manifest.

## Public sample for Gate 1

A public 6.31 MiB LISS-IV GeoTIFF sample is available from the `kk947/LISS-IV-Cloud-Removal` Hugging Face Space. The repository includes `scripts/download_liss4_public_sample.py` to fetch it into `data/raw/cloudy/`.

Run:

    python scripts/download_liss4_public_sample.py

This sample is suitable for **format/radiometry/inference smoke tests only**. It is not a verified cloudy/clear ground-truth pair, so it must not be used to report supervised reconstruction metrics.

## Current limitation

No real LISS-IV cloudy/clear training pair is bundled in this repository. The validation script uses synthetic 8x8 data only for software checks. This is deliberate: third-party satellite data should not be committed until its redistribution terms and exact product format are verified.
