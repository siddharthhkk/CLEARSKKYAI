import argparse
import os

import numpy as np
import rasterio
from PIL import Image


def scale(x):
    x = x.astype(np.float32)
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(
        description="Create natural-color, false-color, and NDVI previews from a LISS-IV GeoTIFF."
    )
    ap.add_argument("tif")
    ap.add_argument("--output-dir", default="data/real_preview")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with rasterio.open(args.tif) as src:
        x = src.read().astype(np.float32)

    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 LISS-IV bands [G,R,NIR], got {x.shape}")

    g, r, nir = x

    # Natural color: R,G,B where LISS-IV has no blue; reuse green for B.
    nat = np.stack([scale(r), scale(g), scale(g)], axis=-1)

    # False color: NIR, R, G.
    fcc = np.stack([scale(nir), scale(r), scale(g)], axis=-1)

    ndvi = (nir - r) / (nir + r + 1e-6)
    ndvi_img = np.clip((ndvi + 1.0) * 0.5, 0.0, 1.0)
    ndvi_rgb = np.stack(
        [
            1.0 - ndvi_img,
            ndvi_img,
            1.0 - np.abs(2.0 * ndvi_img - 1.0),
        ],
        axis=-1,
    )

    stem = os.path.splitext(os.path.basename(args.tif))[0]

    for name, arr in [
        ("natural", nat),
        ("fcc", fcc),
        ("ndvi", ndvi_rgb),
    ]:
        Image.fromarray(np.rint(arr * 255).astype(np.uint8)).save(
            os.path.join(args.output_dir, f"{stem}_{name}.png")
        )

    print("=== REAL LISS-IV PREVIEW ===")
    print(f"Input  : {os.path.abspath(args.tif)}")
    print(f"Output : {os.path.abspath(args.output_dir)}")
    print(f"Shape  : {x.shape[1]} x {x.shape[2]}")
    print(f"Written: {stem}_natural.png, {stem}_fcc.png, {stem}_ndvi.png")


if __name__ == "__main__":
    main()
