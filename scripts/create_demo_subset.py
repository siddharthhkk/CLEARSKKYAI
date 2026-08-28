import os
import shutil
import glob
import random

# Configuration
SOURCE_DATA = "data/"
DEMO_DIR = "demo_samples/"
TARGET_PAIRS = 60    # ~120–180 MB total
SEED = 42            # Fixed seed (change or remove for different samples)

def create_demo_subset():
    # Clean out previous demo samples safely inside the function
    if os.path.exists(DEMO_DIR):
        print(f"🧹 Clearing existing {DEMO_DIR} directory...")
        shutil.rmtree(DEMO_DIR)

    os.makedirs(f"{DEMO_DIR}/sar", exist_ok=True)
    os.makedirs(f"{DEMO_DIR}/optical", exist_ok=True)

    print("🔍 Searching for extracted satellite tile pairs...")
    all_sar_files = glob.glob(f"{SOURCE_DATA}/**/s1_*.tif", recursive=True)
    
    if not all_sar_files:
        print("❌ No SAR files found in data/. Ensure extraction has completed!")
        return

    # Shuffle with seed (remove random.seed(SEED) if you want completely fresh samples every run)
    random.seed(SEED)
    random.shuffle(all_sar_files)

    copied_count = 0
    total_bytes = 0

    print(f"📦 Selecting up to {TARGET_PAIRS} diverse tile pairs...")

    for sar_path in all_sar_files:
        if copied_count >= TARGET_PAIRS:
            break

        # Swap BOTH the folder path (/S1/ to /S2/) and the file prefix (s1_ to s2_)
        opt_path = sar_path.replace("/S1/", "/S2/").replace("s1_", "s2_")

        if os.path.exists(opt_path):
            sar_dest = os.path.join(DEMO_DIR, "sar", os.path.basename(sar_path))
            opt_dest = os.path.join(DEMO_DIR, "optical", os.path.basename(opt_path))

            shutil.copy(sar_path, sar_dest)
            shutil.copy(opt_path, opt_dest)

            total_bytes += os.path.getsize(sar_path) + os.path.getsize(opt_path)
            copied_count += 1

    total_mb = total_bytes / (1024 * 1024)
    print("--------------------------------------------------")
    print(f"✅ Demo package created successfully!")
    print(f"📊 Total tile pairs: {copied_count}")
    print(f"💾 Total size: {total_mb:.2f} MB")
    print("--------------------------------------------------")

if __name__ == "__main__":
    create_demo_subset()