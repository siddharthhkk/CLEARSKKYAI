import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def run(script, *args, check=True):
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script), *map(str, args)],
        check=check,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def main():
    with tempfile.TemporaryDirectory(prefix="clearsky_liss4_data_prep_") as temp:
        source_path = os.path.join(temp, "clear.tif")
        source_patches = os.path.join(temp, "clear_patches")
        v3r_data = os.path.join(temp, "v3r")

        rng = np.random.default_rng(19)
        image = rng.integers(20, 900, size=(3, 512, 768), dtype=np.uint16)
        profile = {
            "driver": "GTiff",
            "height": image.shape[1],
            "width": image.shape[2],
            "count": 3,
            "dtype": "uint16",
            "crs": "EPSG:32646",
            "transform": from_origin(500000, 3000000, 5, 5),
        }
        with rasterio.open(source_path, "w", **profile) as dst:
            dst.write(image)

        extraction = run(
            "extract_liss4_clear_patches.py",
            source_path,
            "--output",
            source_patches,
            "--patch-size",
            "256",
            "--stride",
            "128",
        )
        assert "Written      : 15" in extraction.stdout
        source_manifest = os.path.join(source_patches, "manifest.csv")

        sparse_patches = run(
            "extract_liss4_clear_patches.py",
            source_path,
            "--output",
            os.path.join(temp, "sparse_patches"),
            "--patch-size",
            "256",
            "--stride",
            "512",
        )
        assert "Written      : 4" in sparse_patches.stdout

        band_paths = []
        band_profile = {**profile, "count": 1}
        for index, band_name in enumerate(("green", "red", "nir")):
            band_path = os.path.join(temp, f"{band_name}.tif")
            with rasterio.open(band_path, "w", **band_profile) as dst:
                dst.write(image[index], 1)
            band_paths.append(band_path)

        separate_band_patches = os.path.join(temp, "separate_band_patches")
        band_extraction = run(
            "extract_liss4_clear_patches.py",
            "--clear-bands",
            *band_paths,
            "--output",
            separate_band_patches,
            "--patch-size",
            "256",
            "--stride",
            "128",
        )
        assert "Written      : 15" in band_extraction.stdout
        with np.load(os.path.join(source_patches, "sample_000000.npz")) as stacked:
            with np.load(
                os.path.join(separate_band_patches, "sample_000000.npz")
            ) as separate:
                np.testing.assert_array_equal(stacked["clear"], separate["clear"])

        source_train_manifest = os.path.join(source_patches, "source_train.csv")
        source_val_manifest = os.path.join(source_patches, "source_val.csv")
        run(
            "split_liss4_synthetic_manifest.py",
            "--manifest",
            source_manifest,
            "--scene-width",
            "768",
            "--val-fraction",
            "0.40",
            "--train-output",
            source_train_manifest,
            "--val-output",
            source_val_manifest,
        )
        for split_path in (source_train_manifest, source_val_manifest):
            with open(split_path, newline="", encoding="utf-8") as f:
                split_rows = list(csv.DictReader(f))
            assert split_rows
            assert "strict_cloud_proxy_fraction" in split_rows[0]

        shifted_band_path = os.path.join(temp, "misregistered_nir.tif")
        shifted_profile = {
            **band_profile,
            "transform": from_origin(500005, 3000000, 5, 5),
        }
        with rasterio.open(shifted_band_path, "w", **shifted_profile) as dst:
            dst.write(image[2], 1)
        mismatch = run(
            "extract_liss4_clear_patches.py",
            "--clear-bands",
            band_paths[0],
            band_paths[1],
            shifted_band_path,
            "--output",
            os.path.join(temp, "misregistered_patches"),
            check=False,
        )
        assert mismatch.returncode != 0
        assert "identical dimensions, CRS, and affine transform" in mismatch.stderr

        cloud_scene_path = os.path.join(temp, "cloudy_candidate_scene.tif")
        cloud_scene = np.full((3, 512, 512), 100, dtype=np.uint16)
        cloud_scene[:, :256, :256] = 800
        cloud_profile = {**profile, "height": 512, "width": 512}
        with rasterio.open(cloud_scene_path, "w", **cloud_profile) as dst:
            dst.write(cloud_scene)
        screened_patches = os.path.join(temp, "screened_patches")
        screened = run(
            "extract_liss4_clear_patches.py",
            cloud_scene_path,
            "--output",
            screened_patches,
            "--patch-size",
            "256",
            "--stride",
            "256",
            "--max-strict-cloud-proxy-fraction",
            "0.005",
        )
        assert "Written      : 3" in screened.stdout
        assert "Cloud proxy  : 1 patches skipped" in screened.stdout
        with open(
            os.path.join(screened_patches, "manifest.csv"),
            newline="",
            encoding="utf-8",
        ) as f:
            screened_rows = list(csv.DictReader(f))
        assert all(float(row["strict_cloud_proxy_fraction"]) == 0.0 for row in screened_rows)

        incompatible_patch_size = run(
            "extract_liss4_clear_patches.py",
            source_path,
            "--output",
            os.path.join(temp, "wrong_patch_size"),
            "--patch-size",
            "128",
            check=False,
        )
        assert incompatible_patch_size.returncode != 0
        assert "V3R generator requires 256x256" in incompatible_patch_size.stderr

        nodata_path = os.path.join(temp, "clear_one_band_nodata.tif")
        nodata_image = image.copy()
        nodata_image[0, 10, 10] = 0
        nodata_profile = {**profile, "nodata": 0}
        with rasterio.open(nodata_path, "w", **nodata_profile) as dst:
            dst.write(nodata_image)
        nodata_extraction = run(
            "extract_liss4_clear_patches.py",
            nodata_path,
            "--output",
            os.path.join(temp, "nodata_patches"),
            "--patch-size",
            "256",
            "--stride",
            "128",
        )
        assert "Written      : 14" in nodata_extraction.stdout
        assert "Skipped      : 1 invalid/nodata patches" in nodata_extraction.stdout

        collision = run(
            "extract_liss4_clear_patches.py",
            source_path,
            "--output",
            source_patches,
            "--patch-size",
            "256",
            "--stride",
            "128",
            check=False,
        )
        assert collision.returncode != 0
        assert "choose another --output or pass --overwrite" in collision.stderr

        run(
            "make_liss4_synthetic_pretrain_v3r_from_npz.py",
            "--manifest",
            source_manifest,
            "--output",
            v3r_data,
        )
        v3r_manifest = os.path.join(v3r_data, "manifest.csv")
        with open(v3r_manifest, newline="", encoding="utf-8") as f:
            v3r_rows = list(csv.DictReader(f))
        assert len(v3r_rows) == 15
        assert "strict_cloud_proxy_fraction" in v3r_rows[0]
        run(
            "split_liss4_synthetic_manifest.py",
            "--manifest",
            v3r_manifest,
            "--scene-width",
            "768",
            "--val-fraction",
            "0.40",
        )
        validation = run(
            "validate_liss4_synthetic_pretrain.py",
            "--manifest",
            v3r_manifest,
            "--samples-dir",
            v3r_data,
        )
        assert "Samples : 15" in validation.stdout
        assert "PASS: synthetic pretraining dataset is structurally valid." in validation.stdout

        with open(os.path.join(v3r_data, "train_manifest.csv"), newline="", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with open(os.path.join(v3r_data, "val_manifest.csv"), newline="", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))
        assert len(train_rows) == 6
        assert len(val_rows) == 3
        assert all(int(row["col"]) + 256 <= 460 for row in train_rows)
        assert all(int(row["col"]) >= 460 for row in val_rows)

    print("PASS clear GeoTIFF to NPZ patches to V3R synthetic set to spatial split")
    print("PASS stacked and separate-band extraction agree; misregistered bands rejected")
    print("PASS sparse stride above patch size is supported for spaced sampling")
    print("PASS optional bright-neutral cloud heuristic screens contaminated patches")
    print("PASS split separation and generated dataset validation")
    print("PASS fixed V3R patch-size contract and per-band nodata filtering")
    print("PASS overwrite guard for existing generated data")


if __name__ == "__main__":
    main()
