import csv
import os
import tempfile

import numpy as np
import rasterio
import torch

from liss4 import denormalize_liss4, normalize_liss4, suggested_dn_max
from liss4_dataset import LISS4PairDataset
from liss4_dsen2cr import LISS4DSen2CR


def write_tif(path, arr, transform):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[-2],
        "width": arr.shape[-1],
        "count": arr.shape[0],
        "dtype": str(arr.dtype),
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


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
    assert suggested_dn_max(x)[0] == 1023.0
    print("PASS normalization + DN-range checks")


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
        tr = rasterio.transform.from_origin(0, 8, 0.00005, 0.00005)

        write_tif(p, a, tr)

        with rasterio.open(p) as src:
            assert src.count == 3
            assert src.width == 8
            assert src.height == 8
            assert src.crs.to_epsg() == 4326
            assert np.allclose(src.res, (0.00005, 0.00005))

    print("PASS GeoTIFF read/write metadata check")


def check_dataset():
    with tempfile.TemporaryDirectory() as td:
        cloudy_p = os.path.join(td, "cloudy.tif")
        clear_p = os.path.join(td, "clear.tif")
        mask_p = os.path.join(td, "mask.tif")
        vv_p = os.path.join(td, "vv.tif")
        vh_p = os.path.join(td, "vh.tif")
        manifest = os.path.join(td, "manifest.csv")

        tr = rasterio.transform.from_origin(0, 8, 1.0, 1.0)
        x = np.full((3, 8, 8), 100, dtype=np.uint16)
        y = np.full((3, 8, 8), 200, dtype=np.uint16)
        m = np.zeros((1, 8, 8), dtype=np.uint8)
        m[:, 2:6, 2:6] = 1
        vv = np.full((1, 8, 8), -10, dtype=np.float32)
        vh = np.full((1, 8, 8), -15, dtype=np.float32)

        write_tif(cloudy_p, x, tr)
        write_tif(clear_p, y, tr)
        write_tif(mask_p, m, tr)
        write_tif(vv_p, vv, tr)
        write_tif(vh_p, vh, tr)

        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=("cloudy", "clear", "aoi", "split", "mask", "sar_vv", "sar_vh"),
            )
            w.writeheader()
            w.writerow(
                {
                    "cloudy": cloudy_p,
                    "clear": clear_p,
                    "aoi": "test-aoi",
                    "split": "train",
                    "mask": mask_p,
                    "sar_vv": vv_p,
                    "sar_vh": vh_p,
                }
            )

        ds = LISS4PairDataset(manifest, split="train", dn_max=1023.0, use_sar=True)
        item = ds[0]

        assert len(ds) == 1
        assert item["cloudy"].shape == (3, 8, 8)
        assert item["clear"].shape == (3, 8, 8)
        assert item["mask"].shape == (1, 8, 8)
        assert item["sar"].shape == (2, 8, 8)
        assert float(item["has_mask"]) == 1.0
        assert item["aoi"] == "test-aoi"
        print("PASS manifest + paired GeoTIFF dataset check")

        bad_clear = os.path.join(td, "bad_clear.tif")
        bad_tr = rasterio.transform.from_origin(0.1, 8, 1.0, 1.0)
        write_tif(bad_clear, y, bad_tr)

        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=("cloudy", "clear", "aoi", "split", "mask", "sar_vv", "sar_vh"),
            )
            w.writeheader()
            w.writerow(
                {
                    "cloudy": cloudy_p,
                    "clear": bad_clear,
                    "aoi": "bad-aoi",
                    "split": "train",
                    "mask": "",
                    "sar_vv": "",
                    "sar_vh": "",
                }
            )

        bad_ds = LISS4PairDataset(manifest, split="train", dn_max=1023.0, use_sar=False)
        try:
            bad_ds[0]
        except ValueError as exc:
            assert "not identical" in str(exc)
            print("PASS mismatched-grid guard")
        else:
            raise AssertionError("Dataset accepted a mismatched cloudy/clear grid.")


def main():
    check_normalization()
    check_model(False)
    check_model(True)
    check_geotiff_metadata()
    check_dataset()
    print("ALL PHASE-1 SETUP CHECKS PASSED")


if __name__ == "__main__":
    main()
