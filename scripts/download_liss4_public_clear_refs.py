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

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="space",
        allow_patterns=[
            "data/raw/clear/*.tif",
            "data/raw/clear/*.TIF",
        ],
        local_dir=out,
    )

    # snapshot_download preserves repository-relative directories. Flatten
    # only the clear-reference files into the project's intended output dir.
    nested_dirs = [
        os.path.join(out, "data", "raw", "clear"),
        os.path.join(out, "data", "raw_samples"),
        os.path.join(out, "data", "raw", "clear", "data", "raw", "clear"),
    ]

    moved = []
    for nested in nested_dirs:
        if not os.path.isdir(nested):
            continue
        for name in os.listdir(nested):
            if not name.lower().endswith((".tif", ".tiff")):
                continue
            src = os.path.join(nested, name)
            dst = os.path.join(out, name)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            if os.path.exists(dst):
                os.remove(src)
            else:
                shutil.move(src, dst)
            moved.append(dst)

    files = sorted(
        os.path.join(out, name)
        for name in os.listdir(out)
        if name.lower().endswith((".tif", ".tiff"))
    )

    if not files:
        raise RuntimeError(
            "The public Space did not expose any clear-reference GeoTIFFs."
        )

    print(f"PASS downloaded {len(files)} clear-reference GeoTIFF(s).")
    for p in files[:20]:
        print(" -", os.path.relpath(p, out))

    # Clean empty nested directories created by the repository layout.
    for nested in nested_dirs:
        if os.path.isdir(nested):
            try:
                os.rmdir(nested)
            except OSError:
                pass


if __name__ == "__main__":
    main()
