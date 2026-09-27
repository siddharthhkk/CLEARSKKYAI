import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def load_inventory(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_scene(rows, scene_id):
    for r in rows:
        if r["scene_id"] == scene_id:
            return r
    raise SystemExit(f"Scene not found: {scene_id}")


def band_path(row, band):
    names = row["tif_files"].split(";")
    for name in names:
        if name.upper() in (f"{band}.TIF", f"{band}.TIFF"):
            return os.path.join(row["folder"], name)
    for name in names:
        if band.lower() in name.lower():
            return os.path.join(row["folder"], name)
    raise SystemExit(f"{band} not found for {row['scene_id']}")


def read_overlap(row, bounds, size):
    out = []

    for band in ("BAND2", "BAND3", "BAND4"):
        with rasterio.open(band_path(row, band)) as src:
            win = from_bounds(*bounds, transform=src.transform)

            h = min(size, max(1, int(round(win.height))))
            w = min(size, max(1, int(round(win.width))))

            out.append(
                src.read(
                    1,
                    window=win,
                    out_shape=(h, w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
            )

    return np.stack(out, axis=0)


def read_prediction(path, size):
    with rasterio.open(path) as src:
        h = min(size, src.height)
        w = min(size, src.width)

        x = src.read(
            out_shape=(src.count, h, w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

    if x.shape[0] != 3:
        raise SystemExit(f"Prediction must have 3 bands, got {x.shape}")

    return x


def stretch(x, lo=2, hi=98):
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x)
    a, b = np.percentile(v, [lo, hi])
    if b <= a:
        return np.clip(x, 0, 1)
    return np.clip((x - a) / (b - a), 0, 1)


def false_color(x):
    # Project convention: [G, R, NIR] -> display [NIR, R, G].
    return np.stack(
        [stretch(x[2]), stretch(x[1]), stretch(x[0])],
        axis=-1,
    )


def difference_view(a, b):
    d = np.mean(np.abs(a - b), axis=0)
    return stretch(d, 2, 99.5), d


def main():
    ap = argparse.ArgumentParser(
        description="Visual diagnostic for the real Bhoonidhi V3R evaluation."
    )
    ap.add_argument("cloudy_scene")
    ap.add_argument("clear_scene")
    ap.add_argument(
        "--prediction",
        default="data/eval/bhoonidhi_v3r_prediction.tif",
    )
    ap.add_argument(
        "--csv",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=1200,
        help="Maximum preview dimension; keeps RAM usage low (default: 1200).",
    )
    ap.add_argument(
        "--out",
        default="previews/bhoonidhi_v3r_diagnostic.png",
    )
    ap.add_argument(
        "--also-save-diff",
        action="store_true",
    )
    args = ap.parse_args()

    rows = load_inventory(os.path.abspath(args.csv))
    cloudy_row = find_scene(rows, args.cloudy_scene)
    clear_row = find_scene(rows, args.clear_scene)

    with rasterio.open(args.prediction) as pred_src:
        bounds = pred_src.bounds

    # IMPORTANT: this script is only a visualization diagnostic. Read a
    # small downsampled view of the common footprint instead of loading the
    # ~283 million-pixel scenes into RAM.
    size = args.size

    cloudy = read_overlap(
        cloudy_row,
        (bounds.left, bounds.bottom, bounds.right, bounds.top),
        size,
    )
    clear = read_overlap(
        clear_row,
        (bounds.left, bounds.bottom, bounds.right, bounds.top),
        size,
    )
    pred = read_prediction(args.prediction, size)

    h = min(pred.shape[1], cloudy.shape[1], clear.shape[1])
    w = min(pred.shape[2], cloudy.shape[2], clear.shape[2])

    pred = pred[:, :h, :w]
    cloudy = cloudy[:, :h, :w]
    clear = clear[:, :h, :w]

    # Normalize all displays to the project 10-bit DN range.
    cloudy_n = np.clip(cloudy / 1023.0, 0, 1)
    clear_n = np.clip(clear / 1023.0, 0, 1)
    pred_n = np.clip(pred / 1023.0, 0, 1)

    pred_err_view, pred_abs = difference_view(pred_n, clear_n)
    cloudy_err_view, cloudy_abs = difference_view(cloudy_n, clear_n)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    ax[0, 0].imshow(false_color(cloudy_n))
    ax[0, 0].set_title(f"Cloudy input — {cloudy_row['date']}")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(false_color(pred_n))
    ax[0, 1].set_title("V3R output")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(false_color(clear_n))
    ax[0, 2].set_title(f"Temporal reference — {clear_row['date']}")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(cloudy_err_view, cmap="magma")
    ax[1, 0].set_title("|Cloudy − reference|")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(pred_err_view, cmap="magma")
    ax[1, 1].set_title("|V3R − reference|")
    ax[1, 1].axis("off")

    improvement = cloudy_abs - pred_abs
    lim = np.percentile(np.abs(improvement), 99.5)
    if lim <= 0:
        lim = 1.0
    ax[1, 2].imshow(improvement, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[1, 2].set_title("Error change: input − V3R")
    ax[1, 2].axis("off")

    fig.suptitle(
        "Bhoonidhi LISS-IV V3R real-pair diagnostic (NIR / R / G)",
        fontsize=15,
    )
    fig.tight_layout()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("=== BHOONIDHI V3R DIAGNOSTIC ===")
    print(f"Cloudy : {cloudy_row['scene_id']}")
    print(f"Clear  : {clear_row['scene_id']}")
    print(f"Output : {out}")
    print(f"Shape  : {w} x {h} (downsampled preview)")
    print(
        "Difference panel: warm/positive means V3R reduced absolute "
        "error relative to the cloudy input; cool/negative means it increased it."
    )

    if args.also_save_diff:
        diff_out = os.path.splitext(out)[0] + "_error_change.npy"
        np.save(diff_out, improvement.astype(np.float32))
        print(f"Diff NPY: {os.path.abspath(diff_out)}")


if __name__ == "__main__":
    main()
