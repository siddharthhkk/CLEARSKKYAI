import argparse
import csv
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Validate LISS-IV extracted paired patches."
    )
    ap.add_argument("manifest")
    args = ap.parse_args()

    manifest = os.path.abspath(args.manifest)
    base = os.path.dirname(manifest)

    with open(manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("FAIL: patch manifest is empty")

    required = {"patch", "aoi", "split", "row", "col"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"FAIL: missing manifest columns {sorted(missing)}")

    coords = set()
    for i, row in enumerate(rows, start=2):
        p = row["patch"].strip()
        p = p if os.path.isabs(p) else os.path.join(base, p)

        if not os.path.isfile(p):
            raise SystemExit(f"FAIL line {i}: missing patch {p}")

        key = (row["aoi"].strip(), int(row["row"]), int(row["col"]))
        if key in coords:
            raise SystemExit(f"FAIL line {i}: duplicate patch coordinate {key}")
        coords.add(key)

        d = np.load(p)
        c = d["cloudy"]
        t = d["clear"]

        if c.shape != (3, 256, 256):
            raise SystemExit(
                f"FAIL line {i}: cloudy shape {c.shape}; expected (3,256,256)"
            )
        if t.shape != (3, 256, 256):
            raise SystemExit(
                f"FAIL line {i}: clear shape {t.shape}; expected (3,256,256)"
            )
        if not np.isfinite(c).all() or not np.isfinite(t).all():
            raise SystemExit(f"FAIL line {i}: NaN/Inf found")
        if c.dtype.kind not in "fiu" or t.dtype.kind not in "fiu":
            raise SystemExit(f"FAIL line {i}: unexpected dtype")

    print("LISS-IV PATCH MANIFEST: PASS")
    print(f"Patches: {len(rows)}")
    print(f"Unique coordinates: {len(coords)}")
    print("All patches are paired, finite, and exactly 3x256x256.")


if __name__ == "__main__":
    main()
