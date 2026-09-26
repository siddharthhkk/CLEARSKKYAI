import argparse
import csv
import os
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Validate the generated synthetic LISS-IV pretraining set."
    )
    ap.add_argument(
        "--manifest",
        default="data/synthetic_pretrain/manifest.csv",
    )
    ap.add_argument(
        "--samples-dir",
        default="data/synthetic_pretrain",
    )
    args = ap.parse_args()

    manifest = os.path.abspath(args.manifest)
    root = os.path.abspath(args.samples_dir)

    if not os.path.isfile(manifest):
        raise FileNotFoundError(manifest)

    with open(manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("FAIL: synthetic manifest is empty")

    required = {"idx", "scene", "row", "col", "cloud_seed"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"FAIL: missing manifest columns {sorted(missing)}")

    scene_counts = Counter()
    for i, row in enumerate(rows):
        sample = os.path.join(root, f"sample_{int(row['idx']):06d}.npz")
        if not os.path.isfile(sample):
            raise SystemExit(f"FAIL: missing sample file: {sample}")

        d = np.load(sample)
        cloudy = d["cloudy"]
        clear = d["clear"]
        mask = d["mask"]

        if cloudy.shape != (3, 256, 256):
            raise SystemExit(
                f"FAIL: {sample} cloudy shape={cloudy.shape}; expected (3,256,256)"
            )
        if clear.shape != (3, 256, 256):
            raise SystemExit(
                f"FAIL: {sample} clear shape={clear.shape}; expected (3,256,256)"
            )
        if mask.shape != (1, 256, 256):
            raise SystemExit(
                f"FAIL: {sample} mask shape={mask.shape}; expected (1,256,256)"
            )
        if not np.isfinite(cloudy).all() or not np.isfinite(clear).all():
            raise SystemExit(f"FAIL: NaN/Inf found in {sample}")
        if set(np.unique(mask)).difference({0.0, 1.0}):
            raise SystemExit(f"FAIL: mask is not binary in {sample}")

        scene_counts[row["scene"]] += 1

    print("=== SYNTHETIC LISS-IV PRETRAIN VALIDATION ===")
    print(f"Samples : {len(rows)}")
    print(f"Scenes  : {len(scene_counts)}")
    for scene, count in sorted(scene_counts.items()):
        print(f" - {count:4d} samples | {scene}")
    print("Shapes  : cloudy/clear=(3,256,256), mask=(1,256,256)")
    print("Values  : finite")
    print("Masks   : binary")
    print("PASS: synthetic pretraining dataset is structurally valid.")
    print(
        "NOTE: this check does not establish real-cloud performance or "
        "scene-level generalization."
    )


if __name__ == "__main__":
    main()
