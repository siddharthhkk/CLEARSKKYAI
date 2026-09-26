import argparse
import os

import numpy as np
from PIL import Image


def scale(x):
    x = x.astype(np.float32)
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())
    y = (x - lo) / max(hi - lo, 1e-6)
    return np.clip(y, 0.0, 1.0)


def save_rgb(arr, path):
    # LISS-IV is [G, R, NIR]; use R-G-NIR for a useful false-color preview.
    rgb = np.stack([scale(arr[1]), scale(arr[0]), scale(arr[2])], axis=-1)
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description="Preview one Synthetic Clouds V2 NPZ sample.")
    ap.add_argument("sample")
    ap.add_argument("--output-dir", default="data/synthetic_v2_preview")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with np.load(args.sample) as d:
        clear = d["clear"]
        cloudy = d["cloudy"]
        mask = d["mask"][0]
        alpha = d["alpha"][0]
        shadow = d["shadow"][0]

    stem = os.path.splitext(os.path.basename(args.sample))[0]

    save_rgb(clear, os.path.join(args.output_dir, stem + "_clear.png"))
    save_rgb(cloudy, os.path.join(args.output_dir, stem + "_cloudy.png"))

    for name, arr in [("mask", mask), ("alpha", alpha), ("shadow", shadow)]:
        Image.fromarray(np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(args.output_dir, stem + "_" + name + ".png")
        )

    print("=== SYNTHETIC LISS-IV V2 PREVIEW ===")
    print(f"Sample : {os.path.abspath(args.sample)}")
    print(f"Output : {os.path.abspath(args.output_dir)}")
    print(f"Clear  : min={clear.min():.1f} max={clear.max():.1f} mean={clear.mean():.1f}")
    print(f"Cloudy : min={cloudy.min():.1f} max={cloudy.max():.1f} mean={cloudy.mean():.1f}")
    print(f"Mask   : coverage={(mask > 0).mean()*100:.1f}%")
    print(f"Alpha  : min={alpha.min():.3f} max={alpha.max():.3f} mean={alpha.mean():.3f}")
    print(f"Shadow : min={shadow.min():.3f} max={shadow.max():.3f} mean={shadow.mean():.3f}")


if __name__ == "__main__":
    main()
