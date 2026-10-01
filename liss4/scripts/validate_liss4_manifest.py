import argparse
import csv
import os
from collections import defaultdict


REQUIRED = ("cloudy", "clear", "aoi", "split")
SPLITS = {"train", "val", "test"}


def resolve(base, p):
    p = (p or "").strip()
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.join(base, p)


def main():
    ap = argparse.ArgumentParser(
        description="Validate a LISS-IV pair manifest before patch extraction/training."
    )
    ap.add_argument("manifest")
    args = ap.parse_args()

    manifest = os.path.abspath(args.manifest)
    base = os.path.dirname(manifest)

    with open(manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Manifest is empty.")

    missing_cols = [c for c in REQUIRED if c not in rows[0]]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    aoi_splits = defaultdict(set)
    problems = []

    for i, row in enumerate(rows, start=2):
        aoi = row["aoi"].strip()
        split = row["split"].strip().lower()
        cloudy = resolve(base, row.get("cloudy"))
        clear = resolve(base, row.get("clear"))

        if not aoi:
            problems.append(f"line {i}: empty aoi")
        if split not in SPLITS:
            problems.append(
                f"line {i}: split={split!r}; expected one of {sorted(SPLITS)}"
            )
        else:
            aoi_splits[aoi].add(split)

        for label, path in (("cloudy", cloudy), ("clear", clear)):
            if not path:
                problems.append(f"line {i}: empty {label} path")
            elif not os.path.isfile(path):
                problems.append(f"line {i}: missing {label} file: {path}")

        for col in ("mask", "sar_vv", "sar_vh"):
            path = resolve(base, row.get(col))
            if path and not os.path.isfile(path):
                problems.append(f"line {i}: missing {col} file: {path}")

    for aoi, splits in sorted(aoi_splits.items()):
        if len(splits) > 1:
            problems.append(
                f"AOI leakage: {aoi!r} appears in multiple splits: {sorted(splits)}"
            )

    if problems:
        print("LISS-IV MANIFEST: FAIL")
        for p in problems:
            print(f" - {p}")
        raise SystemExit(1)

    counts = defaultdict(int)
    for row in rows:
        counts[row["split"].strip().lower()] += 1

    print("LISS-IV MANIFEST: PASS")
    print(f"Rows      : {len(rows)}")
    print(f"Train     : {counts['train']}")
    print(f"Validation: {counts['val']}")
    print(f"Test      : {counts['test']}")
    print(f"Unique AOIs: {len(aoi_splits)}")
    print("No AOI appears across multiple splits.")
    print("All referenced files exist.")


if __name__ == "__main__":
    main()
