# DSen2-CR pretrained weights

The project does not commit the external DSen2-CR checkpoint.

Download the official DSen2-CR SAR + CARL checkpoint from the original authors:

https://drive.google.com/file/d/1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg/view?usp=sharing

Save it locally as:

    weights/model_SARcarl.hdf5

Then convert it to the PyTorch state-dict used by ClearSky-AI:

    python scripts/convert_dsen2cr_weights.py --src weights/model_SARcarl.hdf5 --dst weights/dsen2cr_sar_carl.pth

The converted .pth file is also kept local and is not committed to the repository.

The DSen2-CR architecture is a 13-band Sentinel-2 + 2-channel Sentinel-1 residual network. It is an external pretrained baseline, not the ClearSkyUNet architecture.

Original project:
https://github.com/ameraner/dsen2-cr

Paper:
Meraner et al. (2020), Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion.

License note:
The original DSen2-CR repository is GPL-3.0. ClearSky-AI does not redistribute the external checkpoint in this repository; users should follow the upstream license terms when using the model/code.
