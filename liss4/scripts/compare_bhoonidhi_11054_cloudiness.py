import argparse
import csv
import itertools
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_band(r, b, bounds, n):
    ns = r["tif_files"].split(";")
    p = None

    for x in ns:
        if x.upper() in (f"{b}.TIF", f"{b}.TIFF"):
            p = os.path.join(r["folder"], x)
            break

    if p is None:
        for x in ns:
            if b.lower() in x.lower():
                p = os.path.join(r["folder"], x)
                break

    if p is None:
        raise FileNotFoundError(f"{b} not found in {r['scene_id']}")

    with rasterio.open(p) as s:
        w = from_bounds(*bounds, transform=s.transform)
        return s.read(
            1,
            window=w,
            out_shape=(n, n),
            resampling=Resampling.bilinear,
        ).astype(np.float32)


def ov(a, b):
    l = max(float(a["left"]), float(b["left"]))
    d = max(float(a["bottom"]), float(b["bottom"]))
    r = min(float(a["right"]), float(b["right"]))
    t = min(float(a["top"]), float(b["top"]))

    if r <= l or t <= d:
        return None

    return l, d, r, t


def score(g, r, n):
    g = np.clip(g / 1023.0, 0, 1)
    r = np.clip(r / 1023.0, 0, 1)
    n = np.clip(n / 1023.0, 0, 1)

    vm = (g + r) / 2.0
    vx = np.maximum(g, r)
    wl = 1.0 - (vx - np.minimum(g, r)) / (vm + 1e-6)

    bw = (vm > 0.22) & (wl > 0.82)
    st = (vm > 0.30) & (wl > 0.88)

    sc = (
        0.45 * np.clip((vm - 0.12) / 0.35, 0, 1)
        + 0.35 * np.clip((wl - 0.65) / 0.35, 0, 1)
        + 0.20 * np.clip((n - 0.18) / 0.45, 0, 1)
    )

    return {
        "bw": 100.0 * float(np.mean(bw)),
        "st": 100.0 * float(np.mean(st)),
        "sc": float(np.mean(sc)),
        "g90": float(np.percentile(g, 90)),
        "r90": float(np.percentile(r, 90)),
        "n90": float(np.percentile(n, 90)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare cloudiness proxies across all 110/54 Bhoonidhi scenes."
    )
    ap.add_argument(
        "--csv",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--path", default="110")
    ap.add_argument("--row", default="54")
    ap.add_argument(
        "--scene",
        action="append",
        help="Optional scene IDs to compare. Repeat --scene for each ID.",
    )
    args = ap.parse_args()

    rows = load_csv(os.path.abspath(args.csv))

    if args.scene:
        ids = set(args.scene)
        ss = [r for r in rows if r["scene_id"] in ids]
    else:
        ss = [
            r for r in rows
            if r["path"] == args.path and r["row"] == args.row
        ]

    ss = [r for r in ss if {"BAND2", "BAND3", "BAND4"}.issubset(
        {x.split(".")[0].upper() for x in r["tif_files"].split(";")}
    )]

    if len(ss) < 2:
        raise SystemExit("FAIL: need at least two complete 3-band scenes.")

    print("=== BHOONIDHI 110/54 CLOUDINESS MATRIX ===")
    print(f"Scenes: {len(ss)}")
    print()

    ds = {}

    for a, b in itertools.combinations(ss, 2):
        bo = ov(a, b)
        if bo is None:
            continue

        aa = {}
        bb = {}

        for bnd in ("BAND2", "BAND3", "BAND4"):
            aa[bnd] = read_band(a, bnd, bo, args.size)
            bb[bnd] = read_band(b, bnd, bo, args.size)

        sa = score(aa["BAND2"], aa["BAND3"], aa["BAND4"])
        sb = score(bb["BAND2"], bb["BAND3"], bb["BAND4"])

        ds[a["scene_id"]] = sa
        ds[b["scene_id"]] = sb

        dsc = sb["sc"] - sa["sc"]
        dbw = sb["bw"] - sa["bw"]

        print(
            f"{a['date']} vs {b['date']} | "
            f"{a['scene_id']} ↔ {b['scene_id']}"
        )
        print(
            f"  cloud score: {sa['sc']:.4f} ↔ {sb['sc']:.4f} "
            f"(B-A {dsc:+.4f})"
        )
        print(
            f"  bright-white: {sa['bw']:.2f}% ↔ {sb['bw']:.2f}% "
            f"(B-A {dbw:+.2f} pp)"
        )
        print(
            f"  strict proxy: {sa['st']:.2f}% ↔ {sb['st']:.2f}%"
        )
        print()

    print("=== SCENE SUMMARY ===")
    for r in sorted(ss, key=lambda x: ds.get(x["scene_id"], {}).get("sc", 999)):
        s = ds.get(r["scene_id"])
        if not s:
            continue
        print(
            f"{r['date']} | {r['scene_id']} | "
            f"cloud_score={s['sc']:.4f} | "
            f"bright_white={s['bw']:.2f}% | "
            f"strict={s['st']:.2f}%"
        )

    print()
    print(
        "Lower proxy values indicate fewer bright, spectrally neutral pixels. "
        "This is a heuristic only; it is not a ground-truth cloud mask and "
        "cross-satellite radiometry can affect the comparison."
    )


if __name__ == "__main__":
    main()
