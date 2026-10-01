import csv
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from liss4 import normalize_liss4


class LISS4PairDataset(Dataset):
    """
    Manifest-driven LISS-IV cloudy/clear dataset.

    Manifest columns:
      cloudy,clear,aoi,split

    Optional columns:
      mask,sar_vv,sar_vh

    Paths are resolved relative to the manifest directory.
    """
    required = ("cloudy", "clear", "aoi", "split")

    def __init__(
        self,
        manifest,
        split="train",
        dn_max=1023.0,
        use_sar=False,
        bands=(1, 2, 3),
    ):
        super().__init__()

        self.manifest = os.path.abspath(manifest)
        self.base = os.path.dirname(self.manifest)
        self.split = split
        self.dn_max = float(dn_max)
        self.use_sar = use_sar
        self.bands = tuple(int(b) for b in bands)

        if self.bands != (1, 2, 3):
            raise ValueError(
                "Phase-1 LISS-IV loader expects a 3-band multispectral GeoTIFF "
                "whose file bands are ordered G,R,NIR. Change this only after "
                "inspecting the actual supplied product."
            )

        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")

        cols = set(rows[0].keys())
        missing = [c for c in self.required if c not in cols]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        self.rows = [
            r for r in rows
            if r["split"].strip().lower() == split.lower()
        ]

        if not self.rows:
            raise ValueError(
                f"No rows found for split={split!r} in {self.manifest}"
            )

    def __len__(self):
        return len(self.rows)

    def _path(self, p):
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.join(self.base, p)

    def _read_optical(self, path):
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(
                    f"LISS-IV file must contain at least 3 bands: {path}"
                )
            arr = src.read(list(self.bands)).astype(np.float32)
            meta = {
                "count": src.count,
                "shape": (src.height, src.width),
                "crs": src.crs,
                "transform": src.transform,
                "res": src.res,
                "nodata": src.nodata,
            }
        return arr, meta

    def _read_sar(self, row, shape):
        vv = self._path(row.get("sar_vv", "").strip())
        vh = self._path(row.get("sar_vh", "").strip())

        if not vv or not vh:
            raise ValueError(
                "use_sar=True requires sar_vv and sar_vh columns in the manifest."
            )

        chans = []
        for p in (vv, vh):
            with rasterio.open(p) as src:
                x = src.read(1).astype(np.float32)
                if x.shape != shape:
                    raise ValueError(
                        f"SAR shape {x.shape} does not match LISS-IV shape {shape}: {p}"
                    )
                chans.append(x)

        return torch.from_numpy(np.stack(chans, axis=0))

    @staticmethod
    def _same_grid(cmeta, tmeta):
        return (
            cmeta["shape"] == tmeta["shape"]
            and cmeta["crs"] == tmeta["crs"]
            and cmeta["transform"] == tmeta["transform"]
            and np.allclose(cmeta["res"], tmeta["res"], atol=1e-6)
        )

    def __getitem__(self, idx):
        row = self.rows[idx]

        cp = self._path(row["cloudy"].strip())
        tp = self._path(row["clear"].strip())

        cloudy, cm = self._read_optical(cp)
        clear, tm = self._read_optical(tp)

        if not self._same_grid(cm, tm):
            raise ValueError(
                "Cloudy/clear grids are not identical. Reprojection/co-registration "
                f"must be completed before training. Cloudy={cp} Clear={tp}"
            )

        cloudy = normalize_liss4(cloudy, self.dn_max)
        clear = normalize_liss4(clear, self.dn_max)

        out = {
            "cloudy": cloudy,
            "clear": clear,
            "mask": torch.zeros(
                1, cloudy.shape[-2], cloudy.shape[-1], dtype=torch.float32
            ),
            "has_mask": torch.tensor(0.0, dtype=torch.float32),
            "aoi": row["aoi"].strip(),
        }

        mp = self._path(row.get("mask", "").strip())
        if mp:
            with rasterio.open(mp) as src:
                m = src.read(1).astype(np.float32)
            if m.shape != cloudy.shape[-2:]:
                raise ValueError(
                    f"Cloud mask shape {m.shape} does not match image "
                    f"{cloudy.shape[-2:]}: {mp}"
                )
            out["mask"] = torch.from_numpy((m > 0).astype(np.float32))[None]
            out["has_mask"] = torch.tensor(1.0, dtype=torch.float32)

        if self.use_sar:
            out["sar"] = self._read_sar(
                row,
                cloudy.shape[-2:],
            )
        else:
            out["sar"] = torch.zeros(
                2, cloudy.shape[-2], cloudy.shape[-1], dtype=torch.float32
            )

        return out
