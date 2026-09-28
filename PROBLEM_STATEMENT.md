# Project problem statement

## Working title

Cloud-affected Sentinel-2 reconstruction with Sentinel-1 SAR

## Problem

Clouds and haze can obscure useful surface information in optical satellite imagery. This personal research project demonstrates whether a published multimodal model can estimate a clearer Sentinel-2 multispectral image from a cloudy Sentinel-2 observation and a spatially co-registered Sentinel-1 radar observation.

The project is scoped to the input contract of a published SAR-guided Sentinel reconstruction model. Earlier work explored a custom LISS-IV model; it remains documented and runnable as a separate research prototype, but it is no longer the primary demo problem.

## Objective

Build a reproducible local application that accepts aligned Sentinel-2 and Sentinel-1 GeoTIFFs, runs the published SAR + CARL checkpoint, displays the cloudy input beside the model estimate, and exports a 13-band GeoTIFF. Where a paired clear reference exists, the app can also display that reference and calculate patch-level metrics.

This is an integration and evaluation project using an existing pretrained model. The project does not claim to have trained the published checkpoint or reproduced the original paper's training run.

## Inputs and output

- Cloudy Sentinel-2: 13 co-registered bands ordered `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12`, stored as surface-reflectance DN in the expected 0–10000 range.
- Sentinel-1: two co-registered backscatter bands in `VV, VH` order, in dB.
- Output: an estimated 13-band Sentinel-2 image on the optical input grid, written as a GeoTIFF. It is an estimate, not a newly acquired cloud-free observation.

The published model preprocesses optical values by clipping to `[0, 10000]` and dividing by 2000. It clips VV to `[-25, 0]` dB and VH to `[-32.5, 0]` dB, then maps each to `[0, 2]`. Input channels are the 13 optical bands followed by VV and VH. Predictions are clipped to `[0, 5]` in model units and scaled back by 2000.

## Data and evaluation plan

The public SEN12MS-CR dataset supplies co-registered Sentinel-1, cloudy Sentinel-2, and clear Sentinel-2 triplets. Its official full download is 272 GB, so the app supports direct user-provided rasters and two local gallery sizes: a quick 25-case set, or an expanded ~2–3 GB materialized subset containing 96 patches from each of ten held-out test scenes and one train-split scene. The expanded gallery has 960 split-held-out patches and 96 train-split illustrations, plus four curated challenge cases, for 1,060 total. Three challenge patches have public cloud-coverage annotations; one is labelled urban/built-up (IGBP class 13). The current Parquet mirror omits georeferencing, so the urban example is not attributed to a named city.

The gallery test examples are held-out demonstrations from the published split, not an independent dataset or a new benchmark. Train-split examples illustrate the model's training-data family but do not assert that each exact selected patch was used in the released checkpoint's training run. Report per-patch MAE, RMSE, and PSNR only when a matching clear target is supplied. The clear image is a co-registered paired reference and may not be a simultaneous observation, so reference-based metrics are not perfect ground truth. Do not present train-split scores as held-out performance.

## Success criteria

1. Reject missing bands, incorrect band counts, and Sentinel-1/2 grid mismatches before inference.
2. Load the published checkpoint strictly and run finite inference on a 15-channel test input.
3. Preserve the optical input's size, projection, transform, and band order in the output.
4. Offer a demo with training-source, held-out, high-cloud, and urban/built-up-labelled examples, clearly separating splits and stating provenance.
5. Keep the legacy LISS-IV experiments available without implying they use the published Sentinel checkpoint.

## Limitations

- A reconstructed pixel is a model estimate; the model can invent or miss details hidden by clouds.
- The published checkpoint and SEN12MS-CR distribution have their own licensing and citation requirements; follow the original project and dataset terms.
- The expanded gallery covers ten held-out scenes and four seasons from one dataset, but is not independent evidence of cross-dataset or operational generalization.
- The reorganized mirror omits georeferencing, so the demo patches are not map-ready and the urban-labelled patch cannot be tied to a named city without recovering source geospatial metadata.
- Real user imagery must already be co-registered and prepared in the model's expected band order, units, and SAR dB convention.
- No claim is made here that this checkpoint is optimal for Indian imagery, LISS-IV, or every Sentinel processing level.

## Key references

- Meraner et al. (2020), [original DSen2-CR implementation and checkpoint links](https://github.com/ameraner/dsen2-cr).
- Meraner, A., Ebel, P., Zhu, X. X., & Schmitt, M. (2020), “Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion,” *ISPRS Journal of Photogrammetry and Remote Sensing*, 166, 333–346. [DOI](https://doi.org/10.1016/j.isprsjprs.2020.05.013).
- Ebel et al., [official SEN12MS-CR dataset record](https://mediatum.ub.tum.de/1554803) (122,218 triplets; CC BY 4.0).
- [SEN12MS-CR benchmark page](https://patricktum.github.io/cloud_removal/sen12mscr/).
- [SEN12MS-CR-derived cloud coverage annotations](https://zenodo.org/records/17114706).
- [SEN12MS IGBP land-cover labels](https://github.com/schmitt-muc/SEN12MS).
