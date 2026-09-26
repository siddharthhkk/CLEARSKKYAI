import argparse
import os

import rasterio


def main():
    ap = argparse.ArgumentParser(
        description="Validate a directory of native LISS-IV GeoTIFF sources."
    )
    ap.add_argument("directory")
    args = ap.parse_args()

    root = os.path.abspath(args.directory)
    files = sorted(
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(root)
        for name in names
        if name.lower().endswith((".tif", ".tiff"))
    )

    if not files:
        raise SystemExit("FAIL: no GeoTIFF files found")

    print(f"Found {len(files)} GeoTIFF(s)")

    for path in files:
        with rasterio.open(path) as src:
            print(
                f"{os.path.basename(path)} | "
                f"{src.width}x{src.height} | "
                f"bands={src.count} | dtype={src.dtypes} | "
                f"crs={src.crs} | res={src.res}"
            )
            if src.count != 3:
                raise SystemExit(f"FAIL: expected 3 bands: {path}")

    print("PASS: every source is a 3-band GeoTIFF.")


if __name__ == "__main__":
    main()
