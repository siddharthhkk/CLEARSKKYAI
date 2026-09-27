import argparse
import csv
import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling


def load_inventory(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_row(rows, scene_id):
    for r in rows:
        if r["scene_id"] == scene_id:
            return r
    raise SystemExit(f"FAIL: scene not found in inventory: {scene_id}")


def tif_for_band(row, band):
    names = row["tif_files"].split(";")
    for name in names:
        if name.upper().endswith(f"{band}.TIF") or name.upper().endswith(f"{band}.TIFF"):
            return os.path.join(row["folder"], name)

    # Normal Bhoonidhi naming fallback.
    for name in names:
        if name.upper() == f"{band}.TIF":
            return os.path.join(row["folder"], name)

    raise FileNotFoundError(f"{band} not found for scene {row['scene_id']}")


def overlap_bounds(a, b):
    l = max(float(a["left"]), float(b["left"]))
    bt = max(float(a["bottom"]), float(b["bottom"]))
    r = min(float(a["right"]), float(b["right"]))
    t = min(float(a["top"]), float(b["top"]))

    if r <= l or t <= bt:
        raise SystemExit("FAIL: scenes do not overlap.")

    return l, bt, r, t


def read_overlap(scene_path, bounds, out_h=1200, out_w=1200):
    with rasterio.open(scene_path) as src:
        win = from_bounds(*bounds, transform=src.transform)

        arr = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        return arr


def stats(x, q):
    return float(np.percentile(x, q))


def summarize(g, r, n):
    # The public LISS-IV scenes use 10-bit DN in this project.
    scale = 1023.0

    g = np.clip(g / scale, 0, 1)
    r = np.clip(r / scale, 0, 1)
    n = np.clip(n / scale, 0, 1)

    valid = np.isfinite(g) & np.isfinite(r) & np.isfinite(n)
    g = g[valid]
    r = r[valid]
    n = n[valid]

    if g.size == 0:
        raise ValueError("No valid pixels found.")

    vis_mean = (g + r) / 2.0
    vis_max = np.maximum(g, r)
    vis_min = np.minimum(g, r)

    # Bright, spectrally neutral pixels are a simple cloud proxy.
    # This is intentionally only a diagnostic, not a ground-truth cloud mask.
    whiteness = 1.0 - (vis_max - vis_min) / (vis_mean + 1e-6)
    white_bright = (vis_mean > 0.22) & (whiteness > 0.82)

    # A second, stricter proxy catches very bright neutral objects.
    strict = (vis_mean > 0.30) & (whiteness > 0.88)

    # Cloud-like pixels generally have both visible bands elevated.
    both_vis = (g > 0.20) & (r > 0.18)

    score = (
        0.45 * np.clip((vis_mean - 0.12) / 0.35, 0, 1)
        + 0.35 * np.clip((whiteness - 0.65) / 0.35, 0, 1)
        + 0.20 * np.clip((n - 0.18) / 0.45, 0, 1)
    )

    return {
        "valid": int(g.size),
        "g_p50": stats(g, 50),
        "g_p90": stats(g, 90),
        "g_p95": stats(g, 95),
        "g_p99": stats(g, 99),
        "r_p50": stats(r, 50),
        "r_p90": stats(r, 90),
        "r_p95": stats(r, 95),
        "r_p99": stats(r, 99),
        "nir_p50": stats(n, 50),
        "nir_p90": stats(n, 90),
        "nir_p95": stats(n, 95),
        "nir_p99": stats(n, 99),
        "bright_white_pct": float(100 * np.mean(white_bright)),
        "strict_cloud_proxy_pct": float(100 * np.mean(strict)),
        "both_visible_bright_pct": float(100 * np.mean(both_vis)),
        "mean_proxy_score": float(np.mean(score)),
    }


def print_stats(label, s):
    print(f"--- {label} ---")
    print(f"Valid pixels          : {s['valid']}")
    print(
        f"G p50/p90/p95/p99     : "
        f"{s['g_p50']:.3f} / {s['g_p90']:.3f} / "
        f"{s['g_p95']:.3f} / {s['g_p99']:.3f}"
    )
    print(
        f"R p50/p90/p95/p99     : "
        f"{s['r_p50']:.3f} / {s['r_p90']:.3f} / "
        f"{s['r_p95']:.3f} / {s['r_p99']:.3f}"
    )
    print(
        f"NIR p50/p90/p95/p99   : "
        f"{s['nir_p50']:.3f} / {s['nir_p90']:.3f} / "
        f"{s['nir_p95']:.3f} / {s['nir_p99']:.3f}"
    )
    print(f"Bright-white proxy %   : {s['bright_white_pct']:.2f}")
    print(f"Strict cloud proxy %   : {s['strict_cloud_proxy_pct']:.2f}")
    print(f"Both-visible bright %  : {s['both_visible_bright_pct']:.2f}")
    print(f"Mean cloud proxy score : {s['mean_proxy_score']:.4f}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare cloud-like brightness/whiteness statistics for two "
            "overlapping Bhoonidhi LISS-IV scenes."
        )
    )
    ap.add_argument("scene_a")
    ap.add_argument("scene_b")
    ap.add_argument(
        "--csv",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    ap.add_argument("--size", type=int, default=1200)
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    rows = load_inventory(csv_path)

    a = find_row(rows, args.scene_a)
    b = find_row(rows, args.scene_b)

    bounds = overlap_bounds(a, b)

    print("=== BHOONIDHI LISS-IV CLOUDINESS DIAGNOSTIC ===")
    print(f"Scene A : {a['scene_id']}")
    print(f"Date A  : {a['date']}")
    print(f"Scene B : {b['scene_id']}")
    print(f"Date B  : {b['date']}")
    print(
        f"Overlap : left={bounds[0]:.2f}, bottom={bounds[1]:.2f}, "
        f"right={bounds[2]:.2f}, top={bounds[3]:.2f}"
    )
    print()

    rows_by_scene = {}

    for label, row in [("A", a), ("B", b)]:
        g = read_overlap(
            tif_for_band(row, "BAND2"),
            bounds,
            args.size,
            args.size,
        )
        r = read_overlap(
            tif_for_band(row, "BAND3"),
            bounds,
            args.size,
            args.size,
        )
        n = read_overlap(
            tif_for_band(row, "BAND4"),
            bounds,
            args.size,
            args.size,
        )

        s = summarize(g, r, n)
        rows_by_scene[label] = s
        print_stats(
            f"Scene {label} ({row['date']}, path/row {row['path']}/{row['row']})",
            s,
        )

    a_score = rows_by_scene["A"]["mean_proxy_score"]
    b_score = rows_by_scene["B"]["mean_proxy_score"]

    print("=== COMPARISON ===")
    print(
        f"Mean cloud-proxy score difference (B - A): "
        f"{b_score - a_score:+.4f}"
    )
    print(
        f"Bright-white proxy difference (B - A): "
        f"{rows_by_scene['B']['bright_white_pct'] - rows_by_scene['A']['bright_white_pct']:+.2f} percentage points"
    )
    print(
        f"Strict proxy difference (B - A): "
        f"{rows_by_scene['B']['strict_cloud_proxy_pct'] - rows_by_scene['A']['strict_cloud_proxy_pct']:+.2f} percentage points"
    )
    print()
    print(
        "Interpretation: larger proxy values indicate more pixels with "
        "bright, spectrally neutral visible reflectance. This is only a "
        "heuristic cloudiness indicator; haze, bright soil, roofs, salt "
        "flats, and other land-cover effects can also trigger it."
    )


if __name__ == "__main__":
    main()
