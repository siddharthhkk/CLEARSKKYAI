import os
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4_dsen2cr import LISS4DSen2CR


def main():
    with tempfile.TemporaryDirectory(prefix="clearsky_liss4_inference_") as temp:
        source_path = os.path.join(temp, "cloudy.tif")
        checkpoint_path = os.path.join(temp, "tiny_checkpoint.pth")
        output_path = os.path.join(temp, "out", "reconstructed.tif")

        rng = np.random.default_rng(7)
        image = rng.integers(0, 1024, size=(3, 23, 29), dtype=np.uint16)
        transform = from_origin(500000, 3000000, 5, 5)
        profile = {
            "driver": "GTiff",
            "height": image.shape[1],
            "width": image.shape[2],
            "count": 3,
            "dtype": "uint16",
            "crs": "EPSG:32646",
            "transform": transform,
        }
        with rasterio.open(source_path, "w", **profile) as dst:
            dst.write(image)
            dst.update_tags(source="inference-smoke-test")
            for index, name in enumerate(("Green", "Red", "NIR"), start=1):
                dst.set_band_description(index, name)

        band_paths = [os.path.join(temp, f"band{index}.tif") for index in (2, 3, 4)]
        band_profile = profile.copy()
        band_profile["count"] = 1
        for index, (path, band_name) in enumerate(zip(band_paths, ("Green", "Red", "NIR"))):
            with rasterio.open(path, "w", **band_profile) as dst:
                dst.write(image[index:index + 1])
                dst.set_band_description(1, band_name)
                dst.update_tags(1, source_band=f"BAND{index + 2}")

        model = LISS4DSen2CR(features=8, blocks=1, res_scale=0.1, use_sar=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "features": 8,
                "blocks": 1,
                "dn_max": 1023.0,
                "epoch": 1,
            },
            checkpoint_path,
        )

        command = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "infer_liss4_real_cloudy_only_overlap.py"),
            "--checkpoint",
            checkpoint_path,
            "--cloudy",
            source_path,
            "--output",
            output_path,
            "--tile",
            "16",
            "--overlap",
            "4",
            "--device",
            "cpu",
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if "Tiles      : 6 (2 rows x 3 cols)" not in result.stdout:
            raise AssertionError(f"Unexpected tile layout:\n{result.stdout}")

        with rasterio.open(output_path) as output:
            assert output.count == 3
            assert output.dtypes == ("uint16", "uint16", "uint16")
            assert (output.height, output.width) == (23, 29)
            assert output.crs.to_epsg() == 32646
            assert output.transform == transform
            assert output.descriptions == ("Green", "Red", "NIR")
            assert output.tags()["source"] == "inference-smoke-test"
            reconstructed = output.read()
            assert np.isfinite(reconstructed).all()
            assert reconstructed.min() >= 0
            assert reconstructed.max() <= 1023

        stacked_bands_output = os.path.join(temp, "out", "separate_bands.tif")
        band_command = command.copy()
        cloudy_index = band_command.index("--cloudy")
        band_command[cloudy_index:cloudy_index + 2] = ["--cloudy-bands", *band_paths]
        band_command[band_command.index("--output") + 1] = stacked_bands_output
        band_result = subprocess.run(
            band_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert "Input      : " in band_result.stdout
        with rasterio.open(stacked_bands_output) as output:
            separate_result = output.read()
            assert output.descriptions == ("Green", "Red", "NIR")
            for index in range(1, 4):
                assert output.tags(index)["source_band"] == f"BAND{index + 1}"
        assert np.array_equal(reconstructed, separate_result)

        overwrite = subprocess.run(
            command[:command.index("--output")]
            + ["--output", source_path]
            + command[command.index("--tile"):],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert overwrite.returncode != 0
        assert "must not overwrite" in overwrite.stderr

        invalid_source = os.path.join(temp, "invalid_two_band.tif")
        invalid_profile = profile.copy()
        invalid_profile["count"] = 2
        with rasterio.open(invalid_source, "w", **invalid_profile) as dst:
            dst.write(image[:2])

        protected_output = os.path.join(temp, "protected.tif")
        protected_contents = b"existing-output-must-survive"
        with open(protected_output, "wb") as f:
            f.write(protected_contents)

        invalid_command = command.copy()
        invalid_command[invalid_command.index("--cloudy") + 1] = invalid_source
        invalid_command[invalid_command.index("--output") + 1] = protected_output
        invalid = subprocess.run(
            invalid_command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert invalid.returncode != 0
        assert "Expected 3-band" in invalid.stderr
        with open(protected_output, "rb") as f:
            assert f.read() == protected_contents

        misaligned_band = os.path.join(temp, "misaligned_band4.tif")
        misaligned_profile = band_profile.copy()
        misaligned_profile["transform"] = from_origin(500005, 3000000, 5, 5)
        with rasterio.open(misaligned_band, "w", **misaligned_profile) as dst:
            dst.write(image[2:3])
        misaligned_command = band_command.copy()
        misaligned_command[misaligned_command.index("--cloudy-bands") + 3] = misaligned_band
        misaligned_command[misaligned_command.index("--output") + 1] = protected_output
        misaligned = subprocess.run(
            misaligned_command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert misaligned.returncode != 0
        assert "identical grid" in misaligned.stderr
        with open(protected_output, "rb") as f:
            assert f.read() == protected_contents

    print("PASS overlap-tiled inference smoke test (CPU, small checkpoint)")
    print("PASS GeoTIFF dimensions, CRS, transform, band descriptions, tags, and DN range")
    print("PASS stacked and separate-band inputs produce identical predictions")
    print("PASS input-overwrite, grid-validation, and output-preservation guards")


if __name__ == "__main__":
    main()
