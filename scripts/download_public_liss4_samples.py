import argparse
import os
import shutil
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download


HF_REPO = "kk947/LISS-IV-Cloud-Removal"
HF_REVISION = "6acd86abe2bc95ca0faa3013f4da0dc4c0d60929"
SPATIAL_URL = (
    "https://storage.googleapis.com/spatialthoughts-public-data/liss4/"
    "RAF20FEB2023032197010000064SSANSTUC00GTDC.zip"
)


def download_spatialthoughts(out_dir):
    raw = os.path.join(out_dir, "spatialthoughts_demo_raw")
    os.makedirs(raw, exist_ok=True)
    zip_path = os.path.join(raw, "liss4_demo.zip")

    print("\n[1/2] Spatial Thoughts native LISS-IV clear scene")
    print(f"URL: {SPATIAL_URL}")
    urllib.request.urlretrieve(SPATIAL_URL, zip_path)

    extract_dir = os.path.join(raw, "extract")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    band_files = {}
    for root, _, names in os.walk(extract_dir):
        for name in names:
            low = name.lower()
            if low in {"band2.tif", "band3.tif", "band4.tif"}:
                band_files[low] = os.path.join(root, name)

    if set(band_files) != {"band2.tif", "band3.tif", "band4.tif"}:
        raise RuntimeError(
            "Spatial Thoughts sample did not contain BAND2.tif, BAND3.tif and BAND4.tif."
        )

    import numpy as np
    import rasterio

    with rasterio.open(band_files["band2.tif"]) as s:
        profile = s.profile.copy()
        profile.update(count=3, dtype="uint16", compress="deflate")
        b2 = s.read(1)
    with rasterio.open(band_files["band3.tif"]) as s:
        b3 = s.read(1)
    with rasterio.open(band_files["band4.tif"]) as s:
        b4 = s.read(1)

    out = os.path.join(out_dir, "spatialthoughts_liss4_clear.tif")
    stacked = np.stack([b2, b3, b4], axis=0)

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(stacked)

    print(f"PASS: saved {out}")
    return out


def download_chennai(out_dir):
    print("\n[2/2] Historical public Chennai LISS-IV cloudy sample")
    print(f"Source: {HF_REPO}@{HF_REVISION}")
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="space",
        revision=HF_REVISION,
        filename="chennai_cloudy_test.tif",
        local_dir=out_dir,
    )

    final = os.path.join(out_dir, "chennai_cloudy_test.tif")
    if os.path.abspath(path) != os.path.abspath(final):
        shutil.copy2(path, final)

    print(f"PASS: saved {final}")
    print(
        "NOTE: this Chennai file has no verified paired clear target in the "
        "public source. Use it for qualitative/out-of-scene tests only."
    )
    return final


def main():
    ap = argparse.ArgumentParser(
        description="Download additional public native LISS-IV samples."
    )
    ap.add_argument("--clear-dir", default="data/raw/clear")
    ap.add_argument("--cloudy-dir", default="data/raw/cloudy")
    args = ap.parse_args()

    os.makedirs(args.clear_dir, exist_ok=True)
    os.makedirs(args.cloudy_dir, exist_ok=True)

    download_spatialthoughts(os.path.abspath(args.clear_dir))
    download_chennai(os.path.abspath(args.cloudy_dir))

    print("\nPUBLIC LISS-IV SAMPLE DOWNLOAD COMPLETE")
    print("The new samples are not treated as additional supervised cloudy/clear pairs.")


if __name__ == "__main__":
    main()
