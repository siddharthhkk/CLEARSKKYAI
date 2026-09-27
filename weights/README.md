# Model weights

Model checkpoints are intentionally kept out of Git.

## Primary LISS-IV model

The active V3R checkpoint is generated locally with:

    python scripts/train_liss4_synthetic.py

A typical local checkpoint path is:

    weights/liss4_dsen2cr_synthetic_v3r.pth

## Published DSen2-CR baseline

The repository contains a PyTorch implementation of the published DSen2-CR architecture, but the external checkpoint is not redistributed here.

Download the official SAR + CARL checkpoint from the original authors and save it locally as:

    weights/model_SARcarl.hdf5

Then convert it with:

    python scripts/convert_dsen2cr_weights.py --src weights/model_SARcarl.hdf5 --dst weights/dsen2cr_sar_carl.pth

Upstream:

    https://github.com/ameraner/dsen2-cr

Paper:

    Meraner et al. (2020), Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion.

Follow the upstream project's license and checkpoint terms when using the published baseline.
