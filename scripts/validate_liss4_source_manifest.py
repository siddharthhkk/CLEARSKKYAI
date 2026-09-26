import argparse
import os

import rasterio


def main():
    ap = argparse.ArgumentParser(
        description="Validate top-level native LISS-IV GeoTIFF sources."
    )
    ap.add_argument("directory")
    args = ap.parse_args()

    root = os.path.abspath(args.directory)
    files = sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.lower().endswith((".tif", ".tiff"))
    )

    if not files:
        raise SystemExit("FAIL: no top-level GeoTIFF source files found")

    print(f"Found {len(files)} top-level GeoTIFF(s)")

    for path in files:
        with rasterio.open(path) as src:
            print(
                f"{os.path.basename(path)} | "
                f"{src.width}x{src.height} | "
                f"bands={src.count} | dtype={src.dtypes} | "
                f"crs={src.crs} | res={src.res}"
            )
            if src.count != 3:
                raise SystemExit(
                    f"FAIL: expected one stacked 3-band LISS-IV GeoTIFF: {path}"
                )

    print("PASS: every top-level source is a 3-band LISS-IV GeoTIFF.")


if __name__ == "__main__":
    main()
