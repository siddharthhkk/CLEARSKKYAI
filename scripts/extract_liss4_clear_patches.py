import argparse
import csv
import os

import numpy as np
import rasterio
from rasterio.windows import Window

V3R_PATCH_SIZE = 256


def patch_starts(length, patch_size, stride):
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract valid clear [G,R,NIR] patches as NPZ files for V3R "
            "synthetic-cloud generation."
        )
    )
    parser.add_argument("clear", help="Three-band clear LISS-IV GeoTIFF in [G,R,NIR] order.")
    parser.add_argument("--output", default="data/synthetic_pretrain_clear")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Patch stride; overlap is allowed to provide more training samples.",
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the generated manifest and matching sample NPZ files.",
    )
    args = parser.parse_args()

    if args.patch_size != V3R_PATCH_SIZE:
        raise ValueError(
            f"--patch-size must be {V3R_PATCH_SIZE}; the V3R generator requires "
            "256x256 source patches."
        )
    if not 1 <= args.stride <= args.patch_size:
        raise ValueError("--stride must be between 1 and --patch-size.")

    clear_path = os.path.abspath(args.clear)
    output_dir = os.path.abspath(args.output)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    os.makedirs(output_dir, exist_ok=True)

    if not args.overwrite:
        conflicts = []
        if os.path.exists(manifest_path):
            conflicts.append(manifest_path)
        with rasterio.open(clear_path) as src:
            existing_starts = patch_starts(src.height, args.patch_size, args.stride)
            col_starts = patch_starts(src.width, args.patch_size, args.stride)
        for idx in range(len(existing_starts) * len(col_starts)):
            sample_path = os.path.join(output_dir, f"sample_{idx:06d}.npz")
            if os.path.exists(sample_path):
                conflicts.append(sample_path)
        if conflicts:
            raise FileExistsError(
                "Generated files already exist; choose another --output or pass "
                "--overwrite. First existing path: " + conflicts[0]
            )

    with rasterio.open(clear_path) as src:
        if src.count != 3:
            raise ValueError(
                f"Expected a stacked 3-band [G,R,NIR] image; got {src.count} bands."
            )

        rows = patch_starts(src.height, args.patch_size, args.stride)
        cols = patch_starts(src.width, args.patch_size, args.stride)
        if not rows or not cols:
            raise ValueError(
                f"Image size {src.width}x{src.height} is smaller than patch size "
                f"{args.patch_size}."
            )

        fields = ("idx", "scene", "row", "col", "cloud_seed")
        rng = np.random.default_rng(args.seed)
        written = 0
        skipped_invalid = 0
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for row in rows:
                for col in cols:
                    window = Window(col, row, args.patch_size, args.patch_size)
                    clear = src.read(
                        indexes=(1, 2, 3),
                        window=window,
                        out_dtype="float32",
                    )
                    band_masks = src.read_masks(
                        indexes=(1, 2, 3),
                        window=window,
                    )
                    all_bands_valid = np.all(band_masks > 0, axis=0)
                    if not np.isfinite(clear).all() or not all_bands_valid.all():
                        skipped_invalid += 1
                        continue

                    cloud_seed = int(rng.integers(0, 2**31 - 1))
                    sample_path = os.path.join(output_dir, f"sample_{written:06d}.npz")
                    np.savez_compressed(
                        sample_path,
                        clear=clear,
                        row=np.int32(row),
                        col=np.int32(col),
                    )
                    writer.writerow(
                        {
                            "idx": written,
                            "scene": clear_path,
                            "row": row,
                            "col": col,
                            "cloud_seed": cloud_seed,
                        }
                    )
                    written += 1

        width, height = src.width, src.height
        crs = src.crs

    if written == 0:
        raise ValueError("No fully valid patches found; check the input nodata mask.")

    print("=== CLEAR LISS-IV PATCH EXTRACTION ===")
    print(f"Source       : {clear_path}")
    print(f"Size         : {width} x {height}")
    print(f"CRS          : {crs}")
    print(f"Patch/stride : {args.patch_size}/{args.stride} px")
    print(f"Written      : {written}")
    print(f"Skipped      : {skipped_invalid} invalid/nodata patches")
    print(f"Output       : {output_dir}")
    print(f"Manifest     : {manifest_path}")
    print("Band order is preserved as supplied; input must be [Green, Red, NIR].")


if __name__ == "__main__":
    main()
