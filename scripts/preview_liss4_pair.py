import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def stretch_pair(a, b):
    out_a = np.empty_like(a, dtype=np.float32)
    out_b = np.empty_like(b, dtype=np.float32)

    for i in range(3):
        v = np.concatenate([a[i].ravel(), b[i].ravel()])
        lo, hi = np.percentile(v, (2, 98))
        if hi <= lo:
            out_a[i] = 0.0
            out_b[i] = 0.0
        else:
            out_a[i] = np.clip((a[i] - lo) / (hi - lo), 0, 1)
            out_b[i] = np.clip((b[i] - lo) / (hi - lo), 0, 1)

    return out_a, out_b


def fcc(x):
    # LISS-IV has no blue band. This is a false-colour composite:
    # R=NIR, G=Red, B=Green.
    return np.transpose(x[[2, 1, 0]], (1, 2, 0))


def main():
    ap = argparse.ArgumentParser(description="Create a visual sanity-check panel for a LISS-IV cloudy/clear pair.")
    ap.add_argument("cloudy")
    ap.add_argument("clear")
    ap.add_argument("--output", default="data/patches/guwahati/pair_preview.png")
    args = ap.parse_args()

    with rasterio.open(args.cloudy) as csrc, rasterio.open(args.clear) as tsrc:
        c = csrc.read().astype(np.float32)
        t = tsrc.read().astype(np.float32)

    if c.shape != t.shape or c.shape[0] != 3:
        raise ValueError(f"Expected both arrays to be [3,H,W], got {c.shape} and {t.shape}")

    cs, ts = stretch_pair(c, t)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for i, name in enumerate(("Green", "Red", "NIR")):
        ax[0, i].imshow(cs[i], cmap="gray")
        ax[0, i].set_title(f"Cloudy {name}")
        ax[1, i].imshow(ts[i], cmap="gray")
        ax[1, i].set_title(f"Clear {name}")

    ax[0, 3].imshow(fcc(cs))
    ax[0, 3].set_title("Cloudy FCC (NIR/R/G)")
    ax[1, 3].imshow(fcc(ts))
    ax[1, 3].set_title("Clear FCC (NIR/R/G)")

    for a in ax.ravel():
        a.axis("off")

    fig.suptitle("LISS-IV Pair Visual Sanity Check", fontsize=14)
    fig.tight_layout()

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview: {out}")


if __name__ == "__main__":
    main()
