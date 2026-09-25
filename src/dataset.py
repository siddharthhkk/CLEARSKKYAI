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

    When is_train=False, the synthetic cloud mask is deterministic so
    validation metrics are stable across epochs.
    """

    def __init__(self, root_dir, is_train=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

        self.sar_files = (
            glob.glob(os.path.join(root_dir, "**/s1_*.tif"), recursive=True)
            + glob.glob(os.path.join(root_dir, "**/s1_*.TIF"), recursive=True)
        )
        self.sar_files.sort()

        self.valid_samples = []
        for sar_path in self.sar_files:
            sar_parts = sar_path.replace("\\", "/").split("/")
            opt_path = "/".join(
                "S2" if part == "S1" else part for part in sar_parts
            )
            opt_path = opt_path.replace("s1_", "s2_")

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

    def _make_cloudy(self, opt_tensor, idx):
        """Create synthetic clouds; deterministic for validation."""
        _, h, w = opt_tensor.shape
        nh = max(1, h // 16)
        nw = max(1, w // 16)

        if self.is_train:
            noise = torch.rand(1, 1, nh, nw)
        else:
            # Stable per-sample validation corruption.
            g = torch.Generator()
            g.manual_seed(42 + idx)
            noise = torch.rand(1, 1, nh, nw, generator=g)

        cloud_mask = F.interpolate(
            noise, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

        cloud_mask = (cloud_mask > 0.65).float()

        return opt_tensor * (1.0 - cloud_mask) + cloud_mask * 1.0

    def __getitem__(self, idx):
        sar_path, opt_path = self.valid_samples[idx]

        with rasterio.open(sar_path) as src:
            sar_img = src.read().astype(np.float32)

        # Sentinel-2: [B04, B03, B02, B08] = [R, G, B, NIR]
        with rasterio.open(opt_path) as src:
            if src.count < 8:
                raise ValueError(
                    f"Expected at least 8 Sentinel-2 bands in {opt_path}, "
                    f"found {src.count}"
                )
            opt_img = src.read([4, 3, 2, 8]).astype(np.float32)

        sar_tensor = normalize_sar(sar_img)
        opt_tensor = normalize_optical(opt_img)

        target_tensor = opt_tensor.clone()
        cloudy_tensor = self._make_cloudy(opt_tensor, idx)

        return sar_tensor, cloudy_tensor, target_tensor
