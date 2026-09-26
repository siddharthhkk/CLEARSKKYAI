import argparse
import os
import shutil

from huggingface_hub import snapshot_download


REPO_ID = "kk947/LISS-IV-Cloud-Removal"


def main():
    ap = argparse.ArgumentParser(
        description="Download public LISS-IV historical clear references from a Hugging Face Space."
    )
    ap.add_argument(
        "--output",
        default="data/raw/clear",
        help="Local directory for clear-reference GeoTIFFs.",
    )
    args = ap.parse_args()

    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)

    print(f"Source : {REPO_ID}")
    print("Filter : data/raw/clear/*.tif")
    print(f"Output : {out}")

    cache_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="space",
        allow_patterns=[
            "data/raw/clear/*.tif",
            "data/raw/clear/*.TIF",
        ],
        local_dir=out,
        local_dir_use_symlinks=False,
    )

    files = []
    for root, _, names in os.walk(out):
        for name in names:
            if name.lower().endswith((".tif", ".tiff")):
                files.append(os.path.join(root, name))

    files.sort()

    if not files:
        raise RuntimeError(
            "The public Space did not expose any clear-reference GeoTIFFs "
            "through its snapshot. Check the Space files or download them manually."
        )

    print(f"PASS downloaded {len(files)} clear-reference GeoTIFF(s).")
    print(f"Snapshot : {cache_dir}")
    for p in files[:20]:
        print(" -", os.path.relpath(p, out))
    if len(files) > 20:
        print(f" ... and {len(files) - 20} more")


if __name__ == "__main__":
    main()
