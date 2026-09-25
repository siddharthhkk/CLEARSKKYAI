# src/preprocess.py
import numpy as np
import torch

# SEN12MS-CR-TS reference value ranges.
# S1 is provided in dB; S2 is provided as reflectance-like integer values.
SAR_MIN = np.array([-25.0, -32.5], dtype=np.float32)[:, None, None]
SAR_MAX = np.array([0.0, 0.0], dtype=np.float32)[:, None, None]
OPTICAL_MIN = 0.0
OPTICAL_MAX = 10000.0


def normalize_sar(arr: np.ndarray) -> torch.Tensor:
    """Normalize Sentinel-1 VV/VH from fixed physical ranges to [-1, 1]."""
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(
            f"SAR input must have shape [2, H, W], got {arr.shape}"
        )

    x = np.clip(arr, SAR_MIN, SAR_MAX)
    x = (x - SAR_MIN) / (SAR_MAX - SAR_MIN)
    return torch.from_numpy(x * 2.0 - 1.0)


def normalize_optical(arr: np.ndarray) -> torch.Tensor:
    """Normalize Sentinel-2 bands with the fixed [0, 10000] range to [-1, 1]."""
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(
            f"Optical input must have shape [C, H, W], got {arr.shape}"
        )

    x = np.clip(arr, OPTICAL_MIN, OPTICAL_MAX) / OPTICAL_MAX
    return torch.from_numpy(x * 2.0 - 1.0)
