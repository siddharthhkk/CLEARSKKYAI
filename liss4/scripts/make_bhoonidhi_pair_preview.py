import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


BANDS = ["BAND2", "BAND3", "BAND4"]


def load_inventory(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_scene(rows, scene_id):
    for row in rows:
        if row["scene_id"] == scene_id:
            return row
    raise SystemExit(f"FAIL: scene not found: {scene_id}")


def band_path(row, band):
    names = row["tif_files"].split(";")
    for name in names:
        if name.upper() == f"{band}.TIF":
            return os.path.join(row["folder"], name)
        if name.upper() == f"{band}.TIFF":
            return os.path.join(row["folder"], name)
    for name in names:
        if band.lower() in name.lower():
            return os.path.join(row["folder"], name)
    raise SystemExit(f"FAIL: {band} missing in {row['scene_id']}")


def overlap(bounds_a, bounds_b):
    l = max(float(bounds_a["left"]), float(bounds_b["left"]))
    b = max(float(bounds_a["bottom"]), float(bounds_b["bottom"]))
    r = min(float(bounds_a["right"]), float(bounds_b["right"]))
    t = min(float(bounds_a["top"]), float(bounds_b["top"]))
    if r <= l or t <= b:
        raise SystemExit("FAIL: scenes have no overlap.")
    return l, b, r, t


def read_native(path, bounds, out_h, out_w):
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        arr = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        return arr


def percentile_stretch(x, lo=2, hi=98):
    q1, q2 = np.percentile(x[np.isfinite(x)], [lo, hi])
    if q2 <= q1:
        return np.clip(x, 0, 1)
    return np.clip((x - q1) / (q2 - q1), 0, 1)


def make_rgb(g, r, n):
    # LISS-IV order in this project: BAND2=G, BAND3=R, BAND4=NIR.
    # Display false-colour composite as NIR-R-G, useful for vegetation/clouds.
    return np.stack(
        [
            percentile_stretch(n),
            percentile_stretch(r),
            percentile_stretch(g),
        ],
        axis=-1,
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create a side-by-side preview of two overlapping Bhoonidhi "
            "LISS-IV scenes on the common footprint."
        )
    )
    ap.add_argument("scene_a")
    ap.add_argument("scene_b")
    ap.add_argument(
        "--csv",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument(
        "--out",
        default="data/raw/bhoonidhi_liss4/bhoonidhi_pair_preview.png",
    )
    ap.add_argument(
        "--npz",
        default="data/raw/bhoonidhi_liss4/bhoonidhi_pair_overlap.npz",
    )
    args = ap.parse_args()

    rows = load_inventory(os.path.abspath(args.csv))
    a = find_scene(rows, args.scene_a)
    b = find_scene(rows, args.scene_b)
    bounds = overlap(a, b)

    print("=== BHOONIDHI LISS-IV PAIR PREVIEW ===")
    print(f"A: {a['scene_id']} ({a['date']})")
    print(f"B: {b['scene_id']} ({b['date']})")
    print(
        f"Overlap bounds: {bounds[0]:.2f}, {bounds[1]:.2f}, "
        f"{bounds[2]:.2f}, {bounds[3]:.2f}"
    )

    arr_a = {}
    arr_b = {}

    for band in BANDS:
        arr_a[band] = read_native(
            band_path(a, band),
            bounds,
            args.size,
            args.size,
        )
        arr_b[band] = read_native(
            band_path(b, band),
            bounds,
            args.size,
            args.size,
        )

    # Keep raw 10-bit DN arrays for later chip extraction/evaluation.
    np.savez_compressed(
        os.path.abspath(args.npz),
        a_g=arr_a["BAND2"],
        a_r=arr_a["BAND3"],
        a_nir=arr_a["BAND4"],
        b_g=arr_b["BAND2"],
        b_r=arr_b["BAND3"],
        b_nir=arr_b["BAND4"],
        bounds=np.array(bounds, dtype=np.float64),
    )

    a_rgb = make_rgb(arr_a["BAND2"], arr_a["BAND3"], arr_a["BAND4"])
    b_rgb = make_rgb(arr_b["BAND2"], arr_b["BAND3"], arr_b["BAND4"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].imshow(a_rgb)
    ax[0].set_title(f"A: {a['date']} | {a['path']}/{a['row']}")
    ax[0].axis("off")

    ax[1].imshow(b_rgb)
    ax[1].set_title(f"B: {b['date']} | {b['path']}/{b['row']}")
    ax[1].axis("off")

    fig.suptitle(
        "Bhoonidhi LISS-IV common footprint — NIR / R / G",
        fontsize=14,
    )
    fig.tight_layout()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Preview saved : {out}")
    print(f"Overlap NPZ   : {os.path.abspath(args.npz)}")


if __name__ == "__main__":
    main()
