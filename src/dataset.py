# src/dataset.py
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import rasterio

from preprocess import normalize_sar, normalize_optical


class SEN12MSDataset(Dataset):
    """
    PyTorch Dataset for SEN12MS-CR-TS SAR-Optical fusion.

    Input:
      - Sentinel-1: VV, VH
      - Sentinel-2: B04, B03, B02, B08 = RGB + NIR

    Training pairs use a clean optical target and a synthetic-cloud
    version of that same target as the cloudy optical input.
    """

    def __init__(self, root_dir, is_train=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

        # Look for SAR files (supports both extensions).
        self.sar_files = (
            glob.glob(os.path.join(root_dir, "**/s1_*.tif"), recursive=True)
            + glob.glob(os.path.join(root_dir, "**/s1_*.TIF"), recursive=True)
        )

        # Match Sentinel-1 and Sentinel-2 tiles by their shared filename key.
        self.valid_samples = []
        for sar_path in self.sar_files:
            opt_path = sar_path.replace("/S1/", "/S2/").replace("s1_", "s2_")

            if not os.path.exists(opt_path) and opt_path.endswith(".tif"):
                opt_path_alt = opt_path.replace(".tif", ".TIF")
                if os.path.exists(opt_path_alt):
                    opt_path = opt_path_alt

            if os.path.exists(opt_path):
                self.valid_samples.append((sar_path, opt_path))

        print(
            f"Dataset Initialized: Found {len(self.valid_samples)} "
            f"valid image pairs in {root_dir}"
        )

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sar_path, opt_path = self.valid_samples[idx]

        # Sentinel-1: [VV, VH]
        with rasterio.open(sar_path) as src:
            sar_img = src.read().astype(np.float32)

        # Sentinel-2: [B04, B03, B02, B08] = [R, G, B, NIR]
        with rasterio.open(opt_path) as src:
            opt_img = src.read([4, 3, 2, 8]).astype(np.float32)

        # Fixed physical normalization shared with inference.
        sar_tensor = normalize_sar(sar_img)
        opt_tensor = normalize_optical(opt_img)

        # Clean optical image is the ground-truth target.
        target_tensor = opt_tensor.clone()

        # Synthetic cloud generation (training only).
        # Low-resolution noise + bilinear upsampling produces cloud-like blobs.
        _, h, w = opt_tensor.shape
        nh = max(1, h // 16)
        nw = max(1, w // 16)
        noise = torch.rand(1, 1, nh, nw)
        cloud_mask = F.interpolate(
            noise, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Threshold to create opaque cloud regions.
        cloud_mask = (cloud_mask > 0.65).float()

        # Optical tensors are in [-1, 1], so +1 is pure white.
        cloudy_tensor = (
            opt_tensor * (1.0 - cloud_mask) + cloud_mask * 1.0
        )

        return sar_tensor, cloudy_tensor, target_tensor
