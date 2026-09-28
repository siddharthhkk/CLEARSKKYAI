# Checkpoints

## Sentinel-2 SAR-guided reconstruction

`dsen2cr_sar_carl.pth` is the PyTorch state-dict conversion of the original authors' published SAR + CARL pretrained checkpoint. It is included with this repository through Git LFS so the Streamlit demo can run after the LFS assets are fetched. This checkpoint was not trained by ClearSky-AI or the repository owner. Its source, conversion method, paper citation, license, and SHA-256 are recorded in [`dsen2cr_sar_carl.provenance.json`](dsen2cr_sar_carl.provenance.json).

The original Keras HDF5 file is not duplicated here. To reproduce the conversion from the source file:

```powershell
python scripts/convert_dsen2cr_weights.py `
  --src weights/model_SARcarl.hdf5 `
  --dst weights/dsen2cr_sar_carl.pth
```

The PyTorch network is in `src/dsen2cr.py`. Train a new, project-generated checkpoint from a SEN12MS-CR training/validation index with:

```powershell
python train.py `
  --manifest path\to\datasetfilelist.csv `
  --data-root path\to\SEN12MS-CR `
  --output weights\dsen2cr_sar_carl_trained.pth `
  --epochs 8 --batch-size 1 --crop-size 128
```

Training output is kept separate from the published pretrained baseline. The app offers the new checkpoint as a selectable option after it exists locally. The small demo gallery is not a suitable training set: it has only one training-source scene and no validation split.

## Historical LISS-IV prototype

`liss4_dsen2cr_synthetic_v3r.pth` is this project's own PyTorch-trained LISS-IV checkpoint and is included through Git LFS. It is distinct from the published Sentinel-2 checkpoint above, and is used by `app_liss4.py` (not the Sentinel-2 `app.py`). It records epoch 16 and 34.501 dB synthetic validation PSNR on a spatial holdout derived from one clear source scene. These are synthetic-validation results, not broad real-scene accuracy claims. Its SHA-256 and model/data notes are in [`liss4_dsen2cr_synthetic_v3r.provenance.json`](liss4_dsen2cr_synthetic_v3r.provenance.json).

The two-scene fine-tuned candidate remains an experiment and is not the default app checkpoint. To reproduce training of V3R, with its split manifests and generated data available locally, run:

```sh
python scripts/train_liss4_synthetic.py --output weights/liss4_dsen2cr_synthetic_v3r.pth
```

The LISS-IV inference script and app default to `weights/liss4_dsen2cr_synthetic_v3r.pth`. On a fresh clone, run `git lfs pull` to fetch it. Satellite imagery and training data remain local and are not included.
