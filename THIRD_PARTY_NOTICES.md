# Third-party notices

## DSen2-CR model and pretrained checkpoint

The Sentinel model architecture and its published pretrained SAR + CARL checkpoint are from the DSen2-CR project by Andrea Meraner, Patrick Ebel, Xiao Xiang Zhu, and Michael Schmitt. The upstream repository is licensed under GNU GPL version 3; a copy is included in [`LICENSE`](LICENSE). This repository's PyTorch training/data pipeline is a new implementation based on the published model, preprocessing, cloud/shadow detector, and CARL objective. The upstream repository already includes a PyTorch architecture definition; this project does not claim to have authored the model or trained the bundled checkpoint.

The original TensorFlow/Keras trainer is replaced for this project's Sentinel workflow by `train.py` and `src/dsen2cr_training.py`. The bundled `weights/dsen2cr_sar_carl.pth` was converted from the authors' released `model_SARcarl.hdf5` using `scripts/convert_dsen2cr_weights.py`. The PyTorch port and training workflow were modified/written for this project on 2026-09-29; see the source-file notices and checkpoint metadata.

Please cite the original paper when using the model, implementation, or dataset:

> Meraner, A., Ebel, P., Zhu, X. X., & Schmitt, M. (2020). Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion. *ISPRS Journal of Photogrammetry and Remote Sensing*, 166, 333–346. https://doi.org/10.1016/j.isprsjprs.2020.05.013

The sample imagery is derived from SEN12MS-CR and is attributed separately in the README; consult the dataset record for its terms and citation.
