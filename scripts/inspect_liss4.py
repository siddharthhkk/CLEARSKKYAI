import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import rasterio

from liss4 import LISS4_BANDS, suggested_dn_max


def stats(x):
    x = x.astype(np.float32)
    return {
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "p01": float(np.nanpercentile(x, 1)),
        "p50": float(np.nanpercentile(x, 50)),
        "p99": float(np.nanpercentile(x, 99)),
        "zero_pct": float(np.mean(x == 0.0) * 100.0),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Inspect a LISS-IV GeoTIFF before building the training manifest."
    )
    ap.add_argument("path")
    args = ap.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        print("=== LISS-IV INSPECTION ===")
        print(f"File        : {path}")
        print(f"Driver      : {src.driver}")
        print(f"Size        : {src.width} x {src.height}")
        print(f"Bands       : {src.count}")
        print(f"Dtype       : {src.dtypes}")
        print(f"CRS         : {src.crs}")
        print(f"Resolution  : {src.res}")
        print(f"Nodata      : {src.nodata}")
        print(f"Bounds      : {src.bounds}")
        print(f"Transform   : {src.transform}")
        print(f"Descriptions: {src.descriptions}")
        print(f"Expected semantics (multispectral): {LISS4_BANDS}")

        arr = src.read().astype(np.float32)

    if arr.shape[0] != 3:
        print(
            "WARNING: this file is not a 3-band multispectral LISS-IV product. "
            "Do not start training until its band layout is verified."
        )

    for i in range(arr.shape[0]):
        s = stats(arr[i])
        print(
            f"Band {i + 1}: min={s['min']:.3f} max={s['max']:.3f} "
            f"mean={s['mean']:.3f} p01={s['p01']:.3f} "
            f"p50={s['p50']:.3f} p99={s['p99']:.3f} "
            f"zero={s['zero_pct']:.2f}%"
        )

    dn, label = suggested_dn_max(arr)
    print(f"Suggested DN ceiling: {dn:.3f} ({label})")
    print(
        "IMPORTANT: the suggestion is diagnostic only. Use one fixed DN ceiling "
        "for every cloudy/clear pair after verifying the product documentation."
    )


if __name__ == "__main__":
    main()
