"""PyTorch training utilities for the published DSen2-CR architecture.

This is a PyTorch training/data-pipeline port for SEN12MS-CR-style paired
Sentinel-1/Sentinel-2 data. It does not imply that the bundled published
checkpoint was trained by this project.
"""

# Adapted training/data workflow for this project, 2026-09-29; GPL-3.0.

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import rasterio
from scipy import ndimage, signal
import torch
from torch.utils.data import Dataset

S2_MIN = 0.0
S2_MAX = 10000.0
OPTICAL_SCALE = 2000.0
SAR_LIMITS = np.array([[-25.0, 0.0], [-32.5, 0.0]], dtype=np.float32)


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    split: str
    cloudy_path: Path
    sar_path: Path
    clear_path: Path


def read_training_samples(manifest: Path, data_root: Path) -> list[TrainingSample]:
    """Read either this project's manifest.csv or upstream's TSV file list."""
    manifest = manifest.resolve()
    data_root = data_root.resolve()
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        first = stream.readline()

    if "cloudy_s2" in first and "clear_target" in first:
        with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            result = []
            for row in rows:
                split = row.get("split", "").strip().lower()
                if split not in {"train", "val", "validation", "test"}:
                    continue
                result.append(
                    TrainingSample(
                        sample_id=row.get("sample_id", "").strip() or f"row-{len(result)}",
                        split="val" if split in {"val", "validation"} else split,
                        cloudy_path=(manifest.parent / row["cloudy_s2"]).resolve(),
                        sar_path=(manifest.parent / row["sar_s1"]).resolve(),
                        clear_path=(manifest.parent / row["clear_target"]).resolve(),
                    )
                )
        return result

    # Original datasetfilelist.csv rows are tab-delimited records whose first
    # field is: split, SAR folder, clear folder, cloudy folder, image filename.
    result = []
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        for line_number, columns in enumerate(csv.reader(stream, delimiter="\t"), start=1):
            if not columns or not columns[0].strip() or columns[0].lstrip().startswith("#"):
                continue
            fields = [field.strip() for field in columns[0].split(",")]
            if len(fields) < 5 or fields[0] not in {"1", "2", "3"}:
                continue  # header, blank, or optional upstream test-list entry
            split = {"1": "train", "2": "val", "3": "test"}[fields[0]]
            sar_folder, clear_folder, cloudy_folder, filename = fields[1:5]
            result.append(
                TrainingSample(
                    sample_id=filename,
                    split=split,
                    cloudy_path=(data_root / cloudy_folder / filename).resolve(),
                    sar_path=(data_root / sar_folder / filename).resolve(),
                    clear_path=(data_root / clear_folder / filename).resolve(),
                )
            )
    if not result:
        raise ValueError(
            f"No samples found in {manifest}. Expected a project manifest.csv or "
            "the upstream SEN12MS-CR datasetfilelist.csv format."
        )
    return result


def _read_triplet(sample: TrainingSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with (
        rasterio.open(sample.cloudy_path) as cloudy_ds,
        rasterio.open(sample.sar_path) as sar_ds,
        rasterio.open(sample.clear_path) as clear_ds,
    ):
        expected = (13, 2, 13)
        datasets = (cloudy_ds, sar_ds, clear_ds)
        counts = tuple(ds.count for ds in datasets)
        if counts != expected:
            raise ValueError(f"{sample.sample_id}: expected bands S2/SAR/clear {expected}, got {counts}.")
        base_grid = (cloudy_ds.width, cloudy_ds.height, cloudy_ds.crs, cloudy_ds.transform)
        for ds in (sar_ds, clear_ds):
            grid = (ds.width, ds.height, ds.crs, ds.transform)
            if grid != base_grid:
                raise ValueError(f"{sample.sample_id}: SAR, cloudy, and clear rasters are not co-registered.")
        if cloudy_ds.width < 16 or cloudy_ds.height < 16:
            raise ValueError(f"{sample.sample_id}: patch is too small for a training crop.")
        arrays = tuple(ds.read(masked=True) for ds in datasets)
    output = []
    for name, array in zip(("cloudy", "SAR", "clear"), arrays):
        if np.ma.getmaskarray(array).any():
            raise ValueError(f"{sample.sample_id}: {name} raster contains masked/nodata pixels.")
        value = np.asarray(array, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{sample.sample_id}: {name} raster has non-finite values.")
        output.append(value)
    return output[0], output[1], output[2]


def cloud_cloudshadow_mask(optical_dn: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Port the upstream cloud/shadow detector; returns its binary training mask."""
    image = np.clip(optical_dn.astype(np.float32, copy=False) / 10000.0, 0.0, 1.0)
    blue, green, red = image[1], image[2], image[3]
    cirrus, nir, swir1 = image[10], image[7], image[11]

    def rescale(data: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
        return (data - limits[0]) / (limits[1] - limits[0])

    def normalized_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        denominator = left + right
        denominator = np.where(denominator == 0, 0.001, denominator)
        return (left - right) / denominator

    score = np.ones_like(blue, dtype=np.float32)
    score = np.minimum(score, rescale(blue, (0.1, 0.5)))
    score = np.minimum(score, rescale(image[0], (0.1, 0.3)))
    score = np.minimum(score, rescale(image[0] + cirrus, (0.15, 0.2)))
    score = np.minimum(score, rescale(red + green + blue, (0.2, 0.8)))
    ndsi = normalized_difference(green, swir1)
    score = np.minimum(score, rescale(ndsi, (0.8, 0.6)))
    score = ndimage.grey_closing(score, size=(5, 5))
    score = signal.convolve2d(score, np.ones((7, 7), dtype=np.float32) / 49.0, mode="same")
    cloud = np.clip(score, 0.00001, 1.0) >= threshold

    csi = (nir + swir1) / 2.0
    shadow = (csi < csi.min() + 0.75 * (csi.mean() - csi.min())) & (
        blue < blue.min() + (5.0 / 6.0) * (blue.mean() - blue.min())
    )
    shadow = ndimage.median_filter(shadow.astype(np.float32), size=5) > 0
    # The upstream data generator combines both cloud and shadow into the
    # covered class after detecting the signed three-class map.
    return np.logical_or(cloud, shadow).astype(np.float32)[None]


class SEN12MSCRDataset(Dataset):
    def __init__(
        self,
        samples: list[TrainingSample],
        split: str,
        crop_size: int = 128,
        cloud_threshold: float = 0.2,
        augment: bool = False,
    ) -> None:
        self.samples = [sample for sample in samples if sample.split == split]
        self.split = split
        self.crop_size = crop_size
        self.cloud_threshold = cloud_threshold
        self.augment = augment
        if not self.samples:
            raise ValueError(f"No {split!r} samples are listed in the dataset index.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        cloudy, sar, clear = _read_triplet(sample)
        height, width = cloudy.shape[-2:]
        crop = min(self.crop_size, height, width)
        if self.augment:
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        else:
            top, left = (height - crop) // 2, (width - crop) // 2
        sl = np.s_[..., top : top + crop, left : left + crop]
        cloudy, sar, clear = cloudy[sl], sar[sl], clear[sl]
        mask = cloud_cloudshadow_mask(cloudy, self.cloud_threshold)

        if self.augment:
            rotations = random.randrange(4)
            flip_y, flip_x = random.randrange(2), random.randrange(2)
            arrays = [np.rot90(x, rotations, axes=(-2, -1)) for x in (cloudy, sar, clear, mask)]
            if flip_y:
                arrays = [np.flip(x, axis=-2) for x in arrays]
            if flip_x:
                arrays = [np.flip(x, axis=-1) for x in arrays]
            cloudy, sar, clear, mask = arrays

        cloudy = np.clip(cloudy, S2_MIN, S2_MAX) / OPTICAL_SCALE
        clear = np.clip(clear, S2_MIN, S2_MAX) / OPTICAL_SCALE
        lower = SAR_LIMITS[:, 0, None, None]
        upper = SAR_LIMITS[:, 1, None, None]
        sar = (np.clip(sar, lower, upper) - lower) / (upper - lower) * 2.0
        inputs = np.concatenate((cloudy, sar), axis=0).astype(np.float32, copy=False)
        return {
            "inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
            "cloudy": torch.from_numpy(np.ascontiguousarray(cloudy.astype(np.float32))),
            "clear": torch.from_numpy(np.ascontiguousarray(clear.astype(np.float32))),
            "mask": torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32))),
        }


def carl_loss(prediction: torch.Tensor, cloudy: torch.Tensor, clear: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cloud-Adaptive Regularized Loss (CARL) used by the original paper."""
    adaptive = (1.0 - mask) * torch.abs(prediction - cloudy) + mask * torch.abs(prediction - clear)
    return adaptive.mean() + torch.abs(prediction - clear).mean()
