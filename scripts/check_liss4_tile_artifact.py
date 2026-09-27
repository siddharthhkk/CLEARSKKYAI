import argparse
import os

import numpy as np
import rasterio


def block_stats(x, size):
    h, w = x.shape
    nr = h // size
    nc = w // size

    vals = np.zeros((nr, nc), dtype=np.float64)
    for r in range(nr):
        for c in range(nc):
            vals[r, c] = float(
                np.mean(x[r * size:(r + 1) * size, c * size:(c + 1) * size])
            )
    return vals


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Check a LISS-IV GeoTIFF for periodic block/checkerboard artifacts "
            "in the source data or model output."
        )
    )
    ap.add_argument("tif")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--band", type=int, default=1, help="1-based band index.")
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    if args.block <= 0:
        raise ValueError("--block must be positive.")

    with rasterio.open(args.tif) as src:
        if not 1 <= args.band <= src.count:
            raise ValueError(f"Band must be between 1 and {src.count}.")
        x = src.read(args.band).astype(np.float64)

    if x.shape[0] < args.block or x.shape[1] < args.block:
        raise ValueError("Image is smaller than the requested block size.")

    b = block_stats(x, args.block)
    centered = b - b.mean()

    # Checkerboard parity: alternating +/-1 cells.
    parity = np.fromfunction(
        lambda r, c: np.where((r.astype(int) + c.astype(int)) % 2 == 0, 1.0, -1.0),
        b.shape,
    )

    denom = np.linalg.norm(centered) * np.linalg.norm(parity)
    corr = float((centered * parity).sum() / denom) if denom > 0 else 0.0

    even = b[(np.indices(b.shape).sum(axis=0) % 2) == 0]
    odd = b[(np.indices(b.shape).sum(axis=0) % 2) == 1]

    print("=== LISS-IV PERIODIC BLOCK ARTIFACT CHECK ===")
    print(f"Input : {os.path.abspath(args.tif)}")
    print(f"Band  : {args.band}")
    print(f"Size  : {x.shape[1]} x {x.shape[0]}")
    print(f"Block : {args.block} x {args.block}")
    print()
    print(f"All-block mean : {b.mean():.3f} DN")
    print(f"Even blocks    : {even.mean():.3f} DN")
    print(f"Odd blocks     : {odd.mean():.3f} DN")
    print(f"Even-odd diff  : {even.mean() - odd.mean():+.3f} DN")
    print(f"Checker corr   : {corr:+.4f}")
    print()
    print("Block means:")
    for row in b:
        print(" ".join(f"{v:8.2f}" for v in row))

    print()
    print(
        "Large alternating even/odd block differences or checker correlation "
        "indicate a periodic checkerboard/block artifact in the inspected file."
    )


if __name__ == "__main__":
    main()
