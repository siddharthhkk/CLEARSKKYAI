import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class LISS4PatchDataset(Dataset):
    """Dataset for .npz patches produced by extract_liss4_patches.py."""

    def __init__(self, manifest, split="smoke"):
        self.manifest = os.path.abspath(manifest)
        self.base = os.path.dirname(self.manifest)
        self.split = split

        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        required = {"patch", "aoi", "split"}
        if not rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        self.rows = [
            r for r in rows if r["split"].strip().lower() == split.lower()
        ]
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row["patch"].strip()
        if not os.path.isabs(path):
            path = os.path.join(self.base, path)

        d = np.load(path)
        cloudy = d["cloudy"].astype(np.float32)
        clear = d["clear"].astype(np.float32)

        if cloudy.shape != clear.shape or cloudy.ndim != 3 or cloudy.shape[0] != 3:
            raise ValueError(
                f"Bad patch {path}: cloudy={cloudy.shape}, clear={clear.shape}"
            )

        cloudy = np.nan_to_num(cloudy, nan=0.0)
        clear = np.nan_to_num(clear, nan=0.0)

        return {
            "cloudy": torch.from_numpy(cloudy),
            "clear": torch.from_numpy(clear),
            "aoi": row["aoi"].strip(),
            "row": int(row["row"]),
            "col": int(row["col"]),
        }
