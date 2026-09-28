# Model weights

Model checkpoints are intentionally kept out of Git.

## Primary Sentinel DSen2-CR checkpoint

The original authors' full SAR + CARL checkpoint is distributed as `model_SARcarl.hdf5` on [Google Drive](https://drive.google.com/file/d/1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg/view). Download it to this directory, then convert to the local PyTorch state dict:

    python scripts/convert_dsen2cr_weights.py --src weights/model_SARcarl.hdf5 --dst weights/dsen2cr_sar_carl.pth

The Streamlit app and inference script expect `weights/dsen2cr_sar_carl.pth`. Checkpoint files are ignored by Git and are not included in a clone. The model implementation and conversion tool are in this repository; preserve the original project's license and cite Meraner et al. (2020) when presenting results.

Upstream project: https://github.com/ameraner/dsen2-cr

## Historical LISS-IV model

The V3R checkpoint is local and intentionally not redistributed. The best checkpoint currently on the project workstation records epoch 16 and 34.501 dB synthetic validation PSNR. To train/save a checkpoint at the inference command's default path, with the V3R split manifests available, run:

    python scripts/train_liss4_synthetic.py --output weights/liss4_dsen2cr_synthetic_v3r.pth

The legacy LISS-IV inference script's default checkpoint path is:

    weights/liss4_dsen2cr_synthetic_v3r.pth

Checkpoints are ignored by Git; on a fresh clone, create one by training or copy an authorized local checkpoint into this directory.
