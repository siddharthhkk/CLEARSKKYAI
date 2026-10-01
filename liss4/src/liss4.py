import numpy as np
import torch


LISS4_BANDS = ("green", "red", "nir")


def suggested_dn_max(arr):
    """Suggest a DN ceiling without changing the supplied data."""
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        raise ValueError("Cannot infer DN range from an empty array.")

    q = float(np.percentile(a, 99.9))
    mx = float(np.nanmax(a))

    if mx <= 127.5:
        return 127.0, "7-bit-looking"
    if mx <= 1023.0:
        return 1023.0, "10-bit-looking"
    return max(1.0, q), "higher-range/unknown"


def normalize_liss4(arr, dn_max=1023.0):
    """
    Normalize LISS-IV G/R/NIR digital numbers to [0, 1].

    The same fixed dn_max must be used for cloudy and clear images in a pair.
    """
    a = np.asarray(arr, dtype=np.float32)

    if a.ndim not in (3, 4) or a.shape[-3] != 3:
        raise ValueError(
            f"LISS-IV input must have shape [3,H,W] or [B,3,H,W] "
            f"= [G,R,NIR], got {a.shape}"
        )
    if dn_max <= 0:
        raise ValueError(f"dn_max must be positive, got {dn_max}")

    a = np.nan_to_num(a, nan=0.0, posinf=dn_max, neginf=0.0)
    a = np.clip(a, 0.0, float(dn_max)) / float(dn_max)
    return torch.from_numpy(a)


def denormalize_liss4(x, dn_max=1023.0):
    """Convert [0,1] LISS-IV tensors/arrays back to digital numbers."""
    if dn_max <= 0:
        raise ValueError(f"dn_max must be positive, got {dn_max}")

    if isinstance(x, torch.Tensor):
        return torch.clamp(x, 0.0, 1.0) * float(dn_max)

    a = np.asarray(x, dtype=np.float32)
    return np.clip(a, 0.0, 1.0) * float(dn_max)


def check_pair_shapes(cloudy, clear):
    """Basic pair sanity check before training."""
    c = np.asarray(cloudy)
    t = np.asarray(clear)

    if c.shape != t.shape:
        raise ValueError(
            f"Cloudy/clear shapes differ: cloudy={c.shape}, clear={t.shape}"
        )
    if c.ndim != 3 or c.shape[0] != 3:
        raise ValueError(
            f"Expected paired arrays [3,H,W], got {c.shape}"
        )
