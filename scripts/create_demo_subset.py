import argparse
import glob
import os
import random
import shutil


SOURCE_DATA = "data/"
DEMO_DIR = "demo_samples/"
DEFAULT_PAIRS = 60
DEFAULT_SEED = 42

# Same held-out test ROIs used by src/dataset.py.
TEST_ROIS = {
    "ROIs1868/119",
    "ROIs1970/139",
    "ROIs2017/108",
    "ROIs2017/63",
    "ROIs1158/106",
    "ROIs1868/73",
    "ROIs2017/32",
    "ROIs1868/100",
    "ROIs1970/132",
    "ROIs2017/103",
    "ROIs1868/142",
    "ROIs1970/20",
    "ROIs2017/140",
}


def get_roi(path):
    """Extract the dataset ROI label, e.g. ROIs2017/108."""
    path = path.replace("\\", "/")
    parts = path.split("/")

    for marker in ("ROIs1158", "ROIs1868", "ROIs1970", "ROIs2017"):
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts):
                return f"{parts[i]}/{parts[i + 1]}"

    return None


def find_optical_path(sar_path):
    """Map an S1 path to its matching S2 path."""
    path = sar_path.replace("\\", "/")
    opt_path = path.replace("/S1/", "/S2/").replace("s1_", "s2_")

    if os.path.isfile(opt_path):
        return opt_path

    if opt_path.lower().endswith(".tif"):
        alt = opt_path[:-4] + ".TIF"
        if os.path.isfile(alt):
            return alt

    return None


def create_demo_subset(pairs_count=DEFAULT_PAIRS, seed=DEFAULT_SEED, test_only=True):
    if pairs_count <= 0:
        raise ValueError("pairs_count must be greater than 0")

    if not os.path.isdir(SOURCE_DATA):
        print(f"❌ Source dataset not found: {SOURCE_DATA}")
        return

    if os.path.exists(DEMO_DIR):
        print(f"🧹 Clearing existing {DEMO_DIR} directory...")
        shutil.rmtree(DEMO_DIR)

    os.makedirs(os.path.join(DEMO_DIR, "sar"), exist_ok=True)
    os.makedirs(os.path.join(DEMO_DIR, "optical"), exist_ok=True)

    print(f"🔍 Searching {SOURCE_DATA} for Sentinel-1 tiles...")

    sar_files = (
        glob.glob(os.path.join(SOURCE_DATA, "**/s1_*.tif"), recursive=True)
        + glob.glob(os.path.join(SOURCE_DATA, "**/s1_*.TIF"), recursive=True)
    )
    sar_files.sort()

    if not sar_files:
        print("❌ No Sentinel-1 files found in data/. Ensure extraction has completed!")
        return

    candidates = []

    for sar_path in sar_files:
        roi = get_roi(sar_path)

        if test_only and roi not in TEST_ROIS:
            continue

        opt_path = find_optical_path(sar_path)

        if opt_path is not None:
            candidates.append((sar_path, opt_path, roi))

    print(f"📦 Valid S1/S2 pairs found: {len(candidates)}")

    if test_only:
        rois = sorted({roi for _, _, roi in candidates})
        print(f"🧪 Restricting demo selection to {len(rois)} held-out TEST ROIs.")

    if not candidates:
        print("❌ No valid paired samples matched the selected ROI filter.")
        return

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected = candidates[:min(pairs_count, len(candidates))]

    copied_count = 0
    total_bytes = 0
    used_rois = set()

    for sar_path, opt_path, roi in selected:
        sar_dest = os.path.join(DEMO_DIR, "sar", os.path.basename(sar_path))
        opt_dest = os.path.join(DEMO_DIR, "optical", os.path.basename(opt_path))

        shutil.copy2(sar_path, sar_dest)
        shutil.copy2(opt_path, opt_dest)

        total_bytes += os.path.getsize(sar_path) + os.path.getsize(opt_path)
        copied_count += 1
        used_rois.add(roi)

    total_mb = total_bytes / (1024 * 1024)

    print("--------------------------------------------------")
    print("✅ Demo package created successfully!")
    print(f"📊 Tile pairs copied : {copied_count}")
    print(f"🌍 Unique test ROIs  : {len(used_rois)}")
    print(f"🎲 Random seed       : {seed}")
    print(f"💾 Total size        : {total_mb:.2f} MB")
    print("--------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a small ClearSky-AI demo set from SEN12MS-CR-TS."
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help=f"Number of S1/S2 pairs to copy (default: {DEFAULT_PAIRS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for selecting different samples (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--all-rois",
        action="store_true",
        help="Allow sampling from all ROIs instead of held-out test ROIs.",
    )

    args = parser.parse_args()

    create_demo_subset(
        pairs_count=args.pairs,
        seed=args.seed,
        test_only=not args.all_rois,
    )
