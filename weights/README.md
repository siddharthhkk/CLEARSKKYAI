# Model weights

Model checkpoints are intentionally kept out of Git.

## Primary LISS-IV model

The V3R checkpoint is local and intentionally not redistributed. The best checkpoint currently on the project workstation records epoch 16 and 34.501 dB synthetic validation PSNR. To train/save a checkpoint at the inference command's default path, with the V3R split manifests available, run:

    python scripts/train_liss4_synthetic.py --output weights/liss4_dsen2cr_synthetic_v3r.pth

The inference script's default checkpoint path is:

    weights/liss4_dsen2cr_synthetic_v3r.pth

Checkpoints are ignored by Git; on a fresh clone, create one by training or copy an authorized local checkpoint into this directory.

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
