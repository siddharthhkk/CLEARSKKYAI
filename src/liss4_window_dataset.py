"""Windowed paired LISS-IV dataset for supervised cloud-removal training."""

import csv
import os

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from liss4 import normalize_liss4


def _tile_starts(length, patch_size, stride):
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


class LISS4WindowPairDataset(Dataset):
    """Read aligned cloudy/clear GeoTIFF pairs a patch at a time.

    Required CSV fields: cloudy,clear,aoi,split
    Optional field: mask (a one-band cloud mask; nonzero means cloudy)

    Input and target files must have their first three bands ordered
    Green, Red, NIR. Paths are relative to the CSV file unless absolute.
    """

    required = {"cloudy", "clear", "aoi", "split"}

    def __init__(
        self,
        manifest,
        split,
        patch_size=256,
        stride=256,
        dn_max=1023.0,
        augment=False,
    ):
        self.manifest = os.path.abspath(manifest)
        self.base = os.path.dirname(self.manifest)
        self.split = split.strip().lower()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.dn_max = float(dn_max)
        self.augment = bool(augment)

        if self.patch_size < 1:
            raise ValueError("patch_size must be a positive integer")
        if self.stride < 1 or self.stride > self.patch_size:
            raise ValueError("stride must be between 1 and patch_size")
        if self.dn_max <= 0:
            raise ValueError("dn_max must be positive")

        with open(self.manifest, "r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = self.required - columns
            if missing:
                raise ValueError(
                    f"Manifest missing required columns: {sorted(missing)}"
                )
            rows = list(reader)

        rows = [
            row for row in rows
            if (row.get("split") or "").strip().lower() == self.split
        ]
        if not rows:
            raise ValueError(
                f"No rows for split={self.split!r} in {self.manifest}"
            )

        self.rows = []
        self.samples = []
        for row_number, row in enumerate(rows, start=2):
            prepared = dict(row)
            for field in ("cloudy", "clear", "mask"):
                value = (row.get(field) or "").strip()
                prepared[field] = self._resolve(value) if value else ""
            prepared["aoi"] = (row.get("aoi") or "").strip()
            if not prepared["cloudy"] or not prepared["clear"]:
                raise ValueError(
                    f"Manifest row {row_number} needs cloudy and clear paths"
                )
            if not prepared["aoi"]:
                raise ValueError(f"Manifest row {row_number} has an empty aoi")

            cloudy_path = prepared["cloudy"]
            clear_path = prepared["clear"]
            if not os.path.isfile(cloudy_path):
                raise FileNotFoundError(cloudy_path)
            if not os.path.isfile(clear_path):
                raise FileNotFoundError(clear_path)

            with rasterio.open(cloudy_path) as cloudy_src, rasterio.open(
                clear_path
            ) as clear_src:
                self._check_grid(cloudy_src, clear_src, cloudy_path, clear_path)
                if cloudy_src.count < 3 or clear_src.count < 3:
                    raise ValueError(
                        "Cloudy and clear rasters must each have at least three "
                        f"bands: {cloudy_path}, {clear_path}"
                    )
                height, width = cloudy_src.height, cloudy_src.width

            if prepared["mask"]:
                if not os.path.isfile(prepared["mask"]):
                    raise FileNotFoundError(prepared["mask"])
                with rasterio.open(cloudy_path) as cloudy_src, rasterio.open(
                    prepared["mask"]
                ) as mask_src:
                    self._check_grid(
                        cloudy_src, mask_src, cloudy_path, prepared["mask"]
                    )
                    if mask_src.count != 1:
                        raise ValueError(
                            f"Cloud mask must have one band: {prepared['mask']}"
                        )

            row_index = len(self.rows)
            self.rows.append(prepared)
            row_starts = _tile_starts(height, self.patch_size, self.stride)
            col_starts = _tile_starts(width, self.patch_size, self.stride)
            if not row_starts or not col_starts:
                raise ValueError(
                    f"Raster is smaller than patch_size={self.patch_size}: "
                    f"{cloudy_path} ({width}x{height})"
                )
            for top in row_starts:
                for left in col_starts:
                    self.samples.append((row_index, top, left))

        if not self.samples:
            raise ValueError(f"No image patches found for split={self.split!r}")

    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.base, path))

    @staticmethod
    def _check_grid(reference, other, reference_path, other_path):
        same_grid = (
            (reference.height, reference.width) == (other.height, other.width)
            and reference.crs == other.crs
            and reference.transform == other.transform
            and np.allclose(reference.res, other.res, atol=1e-6)
        )
        if not same_grid:
            raise ValueError(
                "Cloudy, clear, and mask rasters must share an identical grid; "
                f"mismatch: {reference_path}, {other_path}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row_index, top, left = self.samples[index]
        row = self.rows[row_index]
        window = Window(left, top, self.patch_size, self.patch_size)

        with rasterio.open(row["cloudy"]) as source:
            cloudy_np = source.read((1, 2, 3), window=window).astype(np.float32)
            cloudy_valid = np.all(
                source.read_masks((1, 2, 3), window=window) > 0, axis=0
            )
        with rasterio.open(row["clear"]) as source:
            clear_np = source.read((1, 2, 3), window=window).astype(np.float32)
            clear_valid = np.all(
                source.read_masks((1, 2, 3), window=window) > 0, axis=0
            )

        valid = (cloudy_valid & clear_valid).astype(np.float32)[None]
        cloudy = normalize_liss4(cloudy_np, self.dn_max)
        clear = normalize_liss4(clear_np, self.dn_max)
        cloud_mask = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)

        if row["mask"]:
            with rasterio.open(row["mask"]) as source:
                mask_np = source.read(1, window=window)
                mask_valid = source.read_masks(1, window=window) > 0
            cloud_mask[0] = ((mask_np > 0) & mask_valid).astype(np.float32)

        valid_tensor = torch.from_numpy(valid)
        cloud_mask_tensor = torch.from_numpy(cloud_mask)
        if self.augment:
            if torch.rand(()) < 0.5:
                cloudy = torch.flip(cloudy, dims=(-1,))
                clear = torch.flip(clear, dims=(-1,))
                valid_tensor = torch.flip(valid_tensor, dims=(-1,))
                cloud_mask_tensor = torch.flip(cloud_mask_tensor, dims=(-1,))
            if torch.rand(()) < 0.5:
                cloudy = torch.flip(cloudy, dims=(-2,))
                clear = torch.flip(clear, dims=(-2,))
                valid_tensor = torch.flip(valid_tensor, dims=(-2,))
                cloud_mask_tensor = torch.flip(cloud_mask_tensor, dims=(-2,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                cloudy = torch.rot90(cloudy, rotations, dims=(-2, -1))
                clear = torch.rot90(clear, rotations, dims=(-2, -1))
                valid_tensor = torch.rot90(valid_tensor, rotations, dims=(-2, -1))
                cloud_mask_tensor = torch.rot90(
                    cloud_mask_tensor, rotations, dims=(-2, -1)
                )

        return {
            "cloudy": cloudy,
            "clear": clear,
            "valid": valid_tensor,
            "mask": cloud_mask_tensor,
            "has_mask": torch.tensor(bool(row["mask"])),
            "aoi": row["aoi"],
        }
