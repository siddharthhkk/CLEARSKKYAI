import argparse
import os
import sys

import numpy as np
import rasterio


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import suggested_dn_max


def inspect(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        return {
            "path": os.path.abspath(path),
            "shape": (src.height, src.width),
            "count": src.count,
            "dtype": tuple(src.dtypes),
            "crs": src.crs,
            "transform": src.transform,
            "res": src.res,
            "bounds": src.bounds,
            "nodata": src.nodata,
            "descriptions": src.descriptions,
            "arr": arr,
        }


def same_grid(a, b):
    return (
        a["shape"] == b["shape"]
        and a["crs"] == b["crs"]
        and a["transform"] == b["transform"]
        and np.allclose(a["res"], b["res"], atol=1e-6)
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare a cloudy and clear LISS-IV GeoTIFF pair before training."
    )
    ap.add_argument("cloudy")
    ap.add_argument("clear")
    args = ap.parse_args()

    for p in (args.cloudy, args.clear):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    c = inspect(args.cloudy)
    t = inspect(args.clear)

    print("=== LISS-IV PAIR CHECK ===")
    for label, x in (("CLOUDY", c), ("CLEAR", t)):
        print(f"\n[{label}]")
        print(f"File        : {x['path']}")
        print(f"Size        : {x['shape'][1]} x {x['shape'][0]}")
        print(f"Bands       : {x['count']}")
        print(f"Dtype       : {x['dtype']}")
        print(f"CRS         : {x['crs']}")
        print(f"Resolution  : {x['res']}")
        print(f"Bounds      : {x['bounds']}")
        print(f"Nodata      : {x['nodata']}")
        print(f"Descriptions: {x['descriptions']}")

        for i in range(min(3, x["count"])):
            band = x["arr"][i]
            print(
                f"Band {i + 1}: min={band.min():.3f} max={band.max():.3f} "
                f"mean={band.mean():.3f} p99={np.percentile(band, 99):.3f}"
            )

    print("\n[PAIR]")
    print(f"Same shape     : {c['shape'] == t['shape']}")
    print(f"Same CRS       : {c['crs'] == t['crs']}")
    print(f"Same transform : {c['transform'] == t['transform']}")
    print(f"Same resolution: {np.allclose(c['res'], t['res'], atol=1e-6)}")
    print(f"Same grid      : {same_grid(c, t)}")

    c_dn, c_label = suggested_dn_max(c["arr"])
    t_dn, t_label = suggested_dn_max(t["arr"])
    print(f"Cloudy DN hint : {c_dn:.3f} ({c_label})")
    print(f"Clear DN hint  : {t_dn:.3f} ({t_label})")

    if same_grid(c, t):
        print("\nPASS: cloudy and clear files share the same raster grid.")
    else:
        print(
            "\nFAIL: cloudy and clear files are not on the same grid. "
            "Do not train until they are explicitly co-registered."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
