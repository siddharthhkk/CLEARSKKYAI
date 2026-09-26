import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser(
        description="Create spatially disjoint train/val manifests for synthetic LISS-IV samples."
    )
    ap.add_argument("--manifest", default="data/synthetic_pretrain_prod/manifest.csv")
    ap.add_argument("--train-output", default=None)
    ap.add_argument("--val-output", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument(
        "--scene-width",
        type=int,
        default=None,
        help=(
            "Source scene width in pixels. Use this when the original source TIFF "
            "is not available locally. For the Spatial Thoughts LISS-IV scene used "
            "for this project, the width is 17733 pixels."
        ),
    )
    args = ap.parse_args()

    if not 0.05 <= args.val_fraction <= 0.40:
        raise ValueError("--val-fraction must be between 0.05 and 0.40")

    if args.scene_width is not None and args.scene_width < 256:
        raise ValueError("--scene-width must be at least 256 pixels")

    manifest = os.path.abspath(args.manifest)
    root = os.path.dirname(manifest)

    with open(manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("FAIL: manifest is empty")

    train = []
    val = []

    scenes = {}
    for r in rows:
        scenes.setdefault(os.path.abspath(r["scene"]), []).append(r)

    for scene, rs in scenes.items():
        if args.scene_width is not None:
            width = args.scene_width
        else:
            try:
                import rasterio

                with rasterio.open(scene) as src:
                    width = src.width
            except Exception as exc:
                raise SystemExit(
                    "FAIL: source scene TIFF is unavailable. "
                    "Provide --scene-width (for this project, 17733)."
                ) from exc

        cut = int(width * (1.0 - args.val_fraction))

        for r in rs:
            col = int(r["col"])
            if col + 256 <= cut:
                train.append(r)
            elif col >= cut:
                val.append(r)
            # Samples crossing the boundary are intentionally discarded.

    if not train or not val:
        raise SystemExit(
            f"FAIL: split produced train={len(train)}, val={len(val)}. "
            "Generate more samples or adjust --val-fraction."
        )

    train_path = os.path.abspath(
        args.train_output or os.path.join(root, "train_manifest.csv")
    )
    val_path = os.path.abspath(
        args.val_output or os.path.join(root, "val_manifest.csv")
    )

    fields = ["idx", "scene", "row", "col", "cloud_seed"]
    for path, subset in [(train_path, train), (val_path, val)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    print("=== SYNTHETIC LISS-IV SPATIAL SPLIT ===")
    print(f"Train samples: {len(train)}")
    print(f"Val samples  : {len(val)}")
    print(f"Val fraction : {args.val_fraction:.2f}")
    print(f"Scene width  : {width}")
    print(f"Train manifest: {train_path}")
    print(f"Val manifest  : {val_path}")
    print("PASS: train/val patches are separated by a spatial scene boundary.")
    print("NOTE: this is synthetic-cloud validation, not a real-cloud benchmark.")


if __name__ == "__main__":
    main()
