import argparse
import csv
import itertools
import os

import rasterio.warp
from pyproj import CRS
from shapely.geometry import box


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def row_bounds(r):
    vals = [
        fnum(r.get("left")),
        fnum(r.get("bottom")),
        fnum(r.get("right")),
        fnum(r.get("top")),
    ]
    if any(v is None for v in vals):
        return None
    l, b, rr, t = vals
    if rr <= l or t <= b:
        return None
    return l, b, rr, t


def scene_geom_in_common_crs(a, b):
    """
    Return both scene rectangles transformed into a common CRS.

    We use the first scene's CRS when it is projected/metres. If the CRSs
    differ, bounds from the second scene are transformed into the first
    scene's CRS before calculating overlap.
    """
    ca = (a.get("crs") or "").strip()
    cb = (b.get("crs") or "").strip()

    ba = row_bounds(a)
    bb = row_bounds(b)

    if not ba or not bb or not ca or not cb:
        return None

    try:
        cra = CRS.from_user_input(ca)
        crb = CRS.from_user_input(cb)
    except Exception:
        return None

    # Scene A stays in its own CRS.
    if cra == crb:
        return box(*ba), box(*bb), cra

    try:
        bb2 = rasterio.warp.transform_bounds(
            crb, cra, *bb, densify_pts=21
        )
        return box(*ba), box(*bb2), cra
    except Exception:
        return None


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Find overlapping Bhoonidhi LISS-IV scenes from the scene "
            "inventory CSV."
        )
    )
    ap.add_argument(
        "csv",
        nargs="?",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=0.50,
        help="Minimum overlap fraction of the smaller scene (default: 0.50).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Show all valid scene pairs, even low-overlap pairs.",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.csv)

    if not os.path.isfile(path):
        raise SystemExit(f"FAIL: inventory CSV not found: {path}")

    rows = load_rows(path)

    if len(rows) < 2:
        raise SystemExit("FAIL: need at least two scenes in the inventory.")

    print("=== BHOONIDHI LISS-IV OVERLAP PAIRS ===")
    print(f"Inventory : {path}")
    print(f"Scenes    : {len(rows)}")
    print(f"Threshold : {args.min_overlap:.0%} of smaller scene")
    print()

    found = []

    for i, j in itertools.combinations(range(len(rows)), 2):
        a = rows[i]
        b = rows[j]

        pair = scene_geom_in_common_crs(a, b)
        if pair is None:
            continue

        ga, gb, crs = pair
        inter = ga.intersection(gb)

        if inter.is_empty or inter.area <= 0:
            overlap_area = 0.0
        else:
            overlap_area = inter.area

        area_a = ga.area
        area_b = gb.area

        if area_a <= 0 or area_b <= 0:
            continue

        frac_a = overlap_area / area_a
        frac_b = overlap_area / area_b

        smaller = min(area_a, area_b)
        overlap_small = overlap_area / smaller if smaller > 0 else 0.0

        union = ga.union(gb).area
        iou = overlap_area / union if union > 0 else 0.0

        if not args.all and overlap_small < args.min_overlap:
            continue

        found.append(
            {
                "a": a,
                "b": b,
                "area_a": area_a,
                "area_b": area_b,
                "overlap": overlap_area,
                "frac_a": frac_a,
                "frac_b": frac_b,
                "small": overlap_small,
                "iou": iou,
                "crs": str(crs),
            }
        )

    found.sort(key=lambda x: x["small"], reverse=True)

    if not found:
        print("No overlapping scene pairs matched the threshold.")
        print("Try --all to inspect every pair.")
        return

    for n, p in enumerate(found, 1):
        a = p["a"]
        b = p["b"]

        print(f"[{n:02d}]")
        print(f"  A: {a['scene_id']}")
        print(f"     date={a['date']} | path/row={a['path']}/{a['row']}")
        print(f"  B: {b['scene_id']}")
        print(f"     date={b['date']} | path/row={b['path']}/{b['row']}")
        print(
            f"  overlap = {p['overlap'] / 1e6:.2f} km² | "
            f"A covered = {p['frac_a']:.2%} | "
            f"B covered = {p['frac_b']:.2%} | "
            f"smaller covered = {p['small']:.2%} | "
            f"IoU = {p['iou']:.2%}"
        )
        print(f"  CRS     = {p['crs']}")
        print()

    print(f"Pairs found: {len(found)}")
    print()
    print(
        "Use the high-overlap pairs as candidates for temporal "
        "clear/cloudy analysis. High overlap alone does not prove that "
        "one scene is clear and the other is cloudy."
    )


if __name__ == "__main__":
    main()
