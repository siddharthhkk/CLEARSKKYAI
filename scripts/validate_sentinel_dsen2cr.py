"""Small integration check for the published Sentinel DSen2-CR checkpoint."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from infer_sentinel_dsen2cr import normalize_inputs, run_inference, validate_rasters


def write_input(path, data, crs="EPSG:32632", transform=None):
    bands, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform or from_origin(500000, 5100000, 10, 10),
    ) as dst:
        dst.write(data)


def main():
    checkpoint = PROJECT_ROOT / "weights" / "dsen2cr_sar_carl.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required local published checkpoint missing: {checkpoint}")

    optical = np.full((13, 33, 40), 2000, dtype=np.uint16)
    sar = np.zeros((2, 33, 40), dtype=np.float32)
    sar[0].fill(-25.0)
    sar[1].fill(-32.5)
    normalized = normalize_inputs(optical.astype(np.float32), sar)
    assert normalized.shape == (15, 33, 40)
    assert np.allclose(normalized[:13], 1.0)
    assert np.allclose(normalized[13:], 0.0)

    with tempfile.TemporaryDirectory(prefix="validate_sentinel_dsen2cr_") as work:
        root = Path(work)
        s2_path, sar_path, output_path = root / "s2.tif", root / "sar.tif", root / "out.tif"
        write_input(s2_path, optical)
        write_input(sar_path, sar)
        validate_rasters(s2_path, sar_path)
        try:
            run_inference(s2_path, sar_path, checkpoint, s2_path, tile=32, overlap=8, device_name="cpu")
        except ValueError as exc:
            assert "different" in str(exc).lower()
        else:
            raise AssertionError("Inference must refuse to overwrite either input raster.")
        run_inference(s2_path, sar_path, checkpoint, output_path, tile=32, overlap=8, device_name="cpu")
        with rasterio.open(output_path) as result:
            assert (result.count, result.width, result.height) == (13, 40, 33)
            assert result.dtypes == ("uint16",) * 13
            assert result.crs.to_epsg() == 32632
            assert result.descriptions[0] == "B01" and result.descriptions[8] == "B8A"
            values = result.read()
            assert np.isfinite(values).all()
            assert values.min() >= 0 and values.max() <= 10000

        misaligned_path = root / "sar_misaligned.tif"
        write_input(misaligned_path, sar, transform=from_origin(500010, 5100000, 10, 10))
        try:
            validate_rasters(s2_path, misaligned_path)
        except ValueError as exc:
            assert "transform" in str(exc).lower()
        else:
            raise AssertionError("A shifted SAR raster should fail co-registration validation.")

    print("Sentinel DSen2-CR integration validation passed (checkpoint load, preprocessing, alignment, GeoTIFF output).")


if __name__ == "__main__":
    main()
