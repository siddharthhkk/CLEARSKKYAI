import argparse
import csv
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import Window


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def same_grid(a, b):
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and a.transform == b.transform
        and np.allclose(a.res, b.res, atol=1e-6)
        and a.count == b.count
    )


def valid_fraction(x):
    return float(np.isfinite(x).all(axis=0).mean())


def main():
    ap = argparse.ArgumentParser(
        description="Extract paired 256x256 LISS-IV cloudy/clear patches."
    )
    ap.add_argument("cloudy")
    ap.add_argument("clear")
    ap.add_argument(
        "--output",
        default="data/patches/guwahati",
        help="Directory for .npz patches and patch_manifest.csv",
    )
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--aoi", default="guwahati_public_sample")
    args = ap.parse_args()

    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("patch-size and stride must be positive")

    cloudy_path = os.path.abspath(args.cloudy)
    clear_path = os.path.abspath(args.clear)
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(cloudy_path) as c, rasterio.open(clear_path) as t:
        if c.count != 3 or t.count != 3:
            raise ValueError("Both LISS-IV files must contain exactly 3 bands.")
        if not same_grid(c, t):
            raise ValueError(
                "Cloudy/clear files are not on the same raster grid. "
                "Run the pair check before extraction."
            )

        h, w = c.height, c.width
        records = []
        saved = 0
        skipped = 0

        for row in range(0, h - args.patch_size + 1, args.stride):
            for col in range(0, w - args.patch_size + 1, args.stride):
                window = Window(col, row, args.patch_size, args.patch_size)

                cloudy = c.read(window=window).astype(np.float32)
                clear = t.read(window=window).astype(np.float32)

                if valid_fraction(cloudy) < 1.0 or valid_fraction(clear) < 1.0:
                    skipped += 1
                    continue

                if not np.isfinite(cloudy).all() or not np.isfinite(clear).all():
                    skipped += 1
                    continue

                name = f"patch_{saved:06d}.npz"
                path = os.path.join(out_dir, name)

                cx, cy = c.transform * (
                    col + args.patch_size / 2.0,
                    row + args.patch_size / 2.0,
                )

                np.savez_compressed(
                    path,
                    cloudy=cloudy,
                    clear=clear,
                    row=np.int32(row),
                    col=np.int32(col),
                    x=np.float64(cx),
                    y=np.float64(cy),
                )

                records.append(
                    {
                        "patch": name,
                        "aoi": args.aoi,
                        "split": "smoke",
                        "row": row,
                        "col": col,
                        "source_cloudy": cloudy_path,
                        "source_clear": clear_path,
                    }
                )
                saved += 1

    manifest = os.path.join(out_dir, "patch_manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        fields = [
            "patch",
            "aoi",
            "split",
            "row",
            "col",
            "source_cloudy",
            "source_clear",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("=== LISS-IV PATCH EXTRACTION ===")
    print(f"Cloudy : {cloudy_path}")
    print(f"Clear  : {clear_path}")
    print(f"Grid   : {w} x {h}")
    print(f"Patch  : {args.patch_size} x {args.patch_size}")
    print(f"Stride : {args.stride}")
    print(f"Saved  : {saved}")
    print(f"Skipped: {skipped}")
    print(f"Output : {out_dir}")
    print(f"Manifest: {manifest}")
    print()
    print(
        "NOTE: split='smoke' is intentionally not a train/val/test split. "
        "With only one scene pair, spatial patches must not be presented as "
        "an independent-scene benchmark."
    )


if __name__ == "__main__":
    main()
