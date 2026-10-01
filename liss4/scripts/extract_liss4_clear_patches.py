import argparse
import csv
import os
from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio.windows import Window

V3R_PATCH_SIZE = 256


class SingleBandStack:
    """Read three co-registered single-band rasters as a [G, R, NIR] stack."""

    def __init__(self, paths):
        self.sources = []
        try:
            for path in paths:
                self.sources.append(rasterio.open(path))
        except Exception:
            self.close()
            raise
        reference = self.sources[0]
        reference_grid = (reference.width, reference.height, reference.crs, reference.transform)
        try:
            for path, src in zip(paths, self.sources):
                if src.count != 1:
                    raise ValueError(f"Expected a single-band raster: {path}")
                grid = (src.width, src.height, src.crs, src.transform)
                if grid != reference_grid:
                    raise ValueError(
                        "The three input bands must have identical dimensions, CRS, "
                        f"and affine transform; mismatch: {path}"
                    )
        except Exception:
            self.close()
            raise

        self.width = reference.width
        self.height = reference.height
        self.crs = reference.crs
        self.transform = reference.transform
        self.count = 3

    def read(self, indexes, window, out_dtype):
        if tuple(indexes) != (1, 2, 3):
            raise ValueError("SingleBandStack only supports reading indexes (1, 2, 3).")
        return np.stack(
            [src.read(1, window=window, out_dtype=out_dtype) for src in self.sources]
        )

    def read_masks(self, indexes, window):
        if tuple(indexes) != (1, 2, 3):
            raise ValueError("SingleBandStack only supports reading indexes (1, 2, 3).")
        return np.stack([src.read_masks(1, window=window) for src in self.sources])

    def close(self):
        for src in getattr(self, "sources", []):
            src.close()


@contextmanager
def open_clear_source(clear_path, clear_band_paths):
    if clear_band_paths:
        stack = SingleBandStack(clear_band_paths)
        try:
            yield stack
        finally:
            stack.close()
    else:
        with rasterio.open(clear_path) as src:
            yield src


def patch_starts(length, patch_size, stride):
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def strict_cloud_proxy_fraction(clear):
    """Fraction of pixels flagged by the project's bright-neutral cloud heuristic."""
    green = np.clip(clear[0] / 1023.0, 0.0, 1.0)
    red = np.clip(clear[1] / 1023.0, 0.0, 1.0)
    mean_visible = (green + red) / 2.0
    whiteness = 1.0 - (
        np.maximum(green, red) - np.minimum(green, red)
    ) / (mean_visible + 1e-6)
    strict_candidate = (mean_visible > 0.30) & (whiteness > 0.88)
    return float(np.mean(strict_candidate))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract valid clear [G,R,NIR] patches as NPZ files for V3R "
            "synthetic-cloud generation."
        )
    )
    parser.add_argument(
        "clear",
        nargs="?",
        help="Stacked clear LISS-IV GeoTIFF in [G,R,NIR] order.",
    )
    parser.add_argument(
        "--clear-bands",
        nargs=3,
        metavar=("GREEN", "RED", "NIR"),
        help="Three co-registered single-band GeoTIFFs in [G,R,NIR] order.",
    )
    parser.add_argument("--output", default="data/synthetic_pretrain_clear")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help=(
            "Patch stride in pixels; values below patch-size overlap, while larger "
            "values sample spaced, non-overlapping patches."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--max-strict-cloud-proxy-fraction",
        type=float,
        default=None,
        help=(
            "Optionally skip patches whose bright-neutral pixel fraction exceeds "
            "this threshold in [0,1]. Heuristic only; it is not a cloud mask."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the generated manifest and matching sample NPZ files.",
    )
    args = parser.parse_args()

    if bool(args.clear) == bool(args.clear_bands):
        raise ValueError("Provide either the stacked positional image or --clear-bands.")

    if args.patch_size != V3R_PATCH_SIZE:
        raise ValueError(
            f"--patch-size must be {V3R_PATCH_SIZE}; the V3R generator requires "
            "256x256 source patches."
        )
    if args.stride < 1:
        raise ValueError("--stride must be at least 1.")
    if args.max_strict_cloud_proxy_fraction is not None and not (
        0.0 <= args.max_strict_cloud_proxy_fraction <= 1.0
    ):
        raise ValueError("--max-strict-cloud-proxy-fraction must be in [0,1].")

    clear_band_paths = (
        [os.path.abspath(path) for path in args.clear_bands]
        if args.clear_bands
        else None
    )
    clear_path = os.path.abspath(args.clear) if args.clear else clear_band_paths[0]
    output_dir = os.path.abspath(args.output)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    if not args.overwrite:
        conflicts = []
        if os.path.exists(manifest_path):
            conflicts.append(manifest_path)
        with open_clear_source(clear_path, clear_band_paths) as src:
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

    with open_clear_source(clear_path, clear_band_paths) as src:
        if src.count != 3:
            raise ValueError(
                f"Expected three [G,R,NIR] bands; got {src.count} bands."
            )

        rows = patch_starts(src.height, args.patch_size, args.stride)
        cols = patch_starts(src.width, args.patch_size, args.stride)
        if not rows or not cols:
            raise ValueError(
                f"Image size {src.width}x{src.height} is smaller than patch size "
                f"{args.patch_size}."
            )

        fields = (
            "idx",
            "scene",
            "row",
            "col",
            "cloud_seed",
            "strict_cloud_proxy_fraction",
        )
        rng = np.random.default_rng(args.seed)
        written = 0
        skipped_invalid = 0
        skipped_cloud_proxy = 0
        os.makedirs(output_dir, exist_ok=True)
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

                    cloud_proxy_fraction = strict_cloud_proxy_fraction(clear)
                    if (
                        args.max_strict_cloud_proxy_fraction is not None
                        and cloud_proxy_fraction
                        > args.max_strict_cloud_proxy_fraction
                    ):
                        skipped_cloud_proxy += 1
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
                            "strict_cloud_proxy_fraction": (
                                f"{cloud_proxy_fraction:.8f}"
                            ),
                        }
                    )
                    written += 1

        width, height = src.width, src.height
        crs = src.crs

    if written == 0:
        raise ValueError("No fully valid patches found; check the input nodata mask.")

    print("=== CLEAR LISS-IV PATCH EXTRACTION ===")
    if clear_band_paths:
        print("Band files   : " + " | ".join(clear_band_paths))
    else:
        print(f"Source       : {clear_path}")
    print(f"Size         : {width} x {height}")
    print(f"CRS          : {crs}")
    print(f"Patch/stride : {args.patch_size}/{args.stride} px")
    print(f"Written      : {written}")
    print(f"Skipped      : {skipped_invalid} invalid/nodata patches")
    print(f"Cloud proxy  : {skipped_cloud_proxy} patches skipped")
    if args.max_strict_cloud_proxy_fraction is not None:
        print(
            "Proxy limit  : "
            f"{args.max_strict_cloud_proxy_fraction:.6f} (heuristic, not cloud mask)"
        )
    print(f"Output       : {output_dir}")
    print(f"Manifest     : {manifest_path}")
    print("Band order is preserved as supplied; input must be [Green, Red, NIR].")


if __name__ == "__main__":
    main()
