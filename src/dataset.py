import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import rasterio
from tqdm import tqdm

from preprocess import normalize_sar, normalize_optical


class SEN12MSDataset(Dataset):
    """
    PyTorch Dataset for SEN12MS-CR-TS SAR-Optical fusion.

    Input:
      - Sentinel-1: VV, VH
      - Sentinel-2: B04, B03, B02, B08 = RGB + NIR

    split:
      - train: official training ROIs
      - val: held-out validation ROIs
      - test: official hold-out test ROIs
      - all: all available ROIs

    Training still uses synthetic clouds over the clean target. This keeps
    the current project's synthetic-cloud training design explicit.
    """

    TEST_ROIS = {
        "ROIs1868/119",
        "ROIs1970/139",
        "ROIs2017/108",
        "ROIs2017/63",
        "ROIs1158/106",
        "ROIs1868/73",
        "ROIs2017/32",
        "ROIs1868/100",
        "ROIs1970/132",
        "ROIs2017/103",
        "ROIs1868/142",
        "ROIs1970/20",
        "ROIs2017/140",
    }

    VAL_ROIS = {
        "ROIs2017/22",
        "ROIs1970/65",
        "ROIs2017/117",
        "ROIs1868/127",
        "ROIs1868/17",
    }

    _PAIR_CACHE = {}

    def __init__(self, root_dir, split="train", transform=None):
        super().__init__()

        if split not in {"train", "val", "test", "all"}:
            raise ValueError(
                f"split must be one of train/val/test/all, got {split}"
            )

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        root_key = os.path.abspath(root_dir)

        if root_key not in self._PAIR_CACHE:
            sar_files = (
                glob.glob(os.path.join(root_dir, "**/s1_*.tif"), recursive=True)
                + glob.glob(os.path.join(root_dir, "**/s1_*.TIF"), recursive=True)
            )
            sar_files.sort()

            pairs = []
            for sar_path in tqdm(
                sar_files,
                desc=f"📂 Indexing SAR/S2 pairs ({root_dir})",
                unit="file",
            ):
                norm_path = sar_path.replace("\\", "/")
                parts = norm_path.split("/")

                roi = None
                for marker in ("ROIs1158", "ROIs1868", "ROIs1970", "ROIs2017"):
                    if marker in parts:
                        i = parts.index(marker)
                        if i + 1 < len(parts):
                            roi = f"{parts[i]}/{parts[i + 1]}"
                        break

                if roi is None:
                    continue

                opt_path = norm_path.replace("/S1/", "/S2/").replace("s1_", "s2_")

                if not os.path.isfile(opt_path) and opt_path.endswith(".tif"):
                    opt_path_alt = opt_path[:-4] + ".TIF"
                    if os.path.isfile(opt_path_alt):
                        opt_path = opt_path_alt

                if os.path.isfile(opt_path):
                    pairs.append((sar_path, opt_path, roi))

            self._PAIR_CACHE[root_key] = pairs
            print(f"✅ Indexed {len(pairs)} valid SAR/S2 pairs.")
        else:
            print(f"⚡ Using cached dataset index for {root_dir}")

        pairs = self._PAIR_CACHE[root_key]

        if self.split == "train":
            self.valid_samples = [
                (sar, opt)
                for sar, opt, roi in pairs
                if roi not in self.TEST_ROIS and roi not in self.VAL_ROIS
            ]
        elif self.split == "val":
            self.valid_samples = [
                (sar, opt)
                for sar, opt, roi in pairs
                if roi in self.VAL_ROIS
            ]
        elif self.split == "test":
            self.valid_samples = [
                (sar, opt)
                for sar, opt, roi in pairs
                if roi in self.TEST_ROIS
            ]
        else:
            self.valid_samples = [(sar, opt) for sar, opt, _ in pairs]

        print(
            f"📦 Dataset Initialized | split={split} | "
            f"{len(self.valid_samples)} samples"
        )

    def __len__(self):
        return len(self.valid_samples)

    def _make_cloudy(self, opt_tensor, idx):
        """Create synthetic clouds; deterministic for validation/test."""
        _, h, w = opt_tensor.shape
        nh = max(1, h // 16)
        nw = max(1, w // 16)

        if self.split == "train":
            noise = torch.rand(1, 1, nh, nw)
        else:
            g = torch.Generator()
            g.manual_seed(42 + idx)
            noise = torch.rand(1, 1, nh, nw, generator=g)

        cloud_mask = F.interpolate(
            noise,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        cloud_mask = (cloud_mask > 0.65).float()

        return opt_tensor * (1.0 - cloud_mask) + cloud_mask * 1.0

    def __getitem__(self, idx):
        sar_path, opt_path = self.valid_samples[idx]

        with rasterio.open(sar_path) as src:
            sar_img = src.read().astype(np.float32)

        # [B04, B03, B02, B08] = [R, G, B, NIR]
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
