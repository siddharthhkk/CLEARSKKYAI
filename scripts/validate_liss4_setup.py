import argparse
import os
import tempfile

import numpy as np
import rasterio
import torch

from liss4 import denormalize_liss4, normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR


def check_normalization():
    x = np.array(
        [
            [[0, 512], [1023, 100]],
            [[10, 500], [900, 1023]],
            [[1023, 0], [256, 768]],
        ],
        dtype=np.float32,
    )

    y = normalize_liss4(x, 1023.0)
    z = denormalize_liss4(y, 1023.0).numpy()

    assert y.shape == (3, 2, 2)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0
    assert np.allclose(z, x, atol=1e-5)
    print("PASS normalization round-trip")


def check_model(use_sar):
    model = LISS4DSen2CR(
        features=32,
        blocks=2,
        res_scale=0.1,
        use_sar=use_sar,
    ).eval()

    cloudy = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        if use_sar:
            sar = torch.rand(2, 2, 32, 32)
            out = model(cloudy, sar)
        else:
            out = model(cloudy)

    assert out.shape == cloudy.shape
    assert torch.isfinite(out).all()
    print(f"PASS model shape/finite check | use_sar={use_sar}")


def check_geotiff_metadata():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.tif")
        a = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)

        profile = {
            "driver": "GTiff",
            "height": 8,
            "width": 8,
            "count": 3,
            "dtype": "uint16",
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_origin(
                0, 8, 0.00005, 0.00005
            ),
        }

        with rasterio.open(p, "w", **profile) as dst:
            dst.write(a)

        with rasterio.open(p) as src:
            assert src.count == 3
            assert src.width == 8
            assert src.height == 8
            assert src.crs.to_epsg() == 4326

    print("PASS GeoTIFF read/write metadata check")


def main():
    ap = argparse.ArgumentParser(
        description="Run Phase-1 LISS-IV pipeline sanity checks."
    )
    ap.parse_args()

    check_normalization()
    check_model(False)
    check_model(True)
    check_geotiff_metadata()

    print("ALL PHASE-1 SETUP CHECKS PASSED")


if __name__ == "__main__":
    main()
