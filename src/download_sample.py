import urllib.request
import os
import sys

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        progress = downloaded / total_size * 100
        sys.stdout.write(f"\rDownloading... {progress:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()

# Create data directory
os.makedirs("data", exist_ok=True)

print("Downloading Sentinel-1 SAR (Radar) for Africa...")
urllib.request.urlretrieve(
    "ftp://m1639953:m1639953@dataserv.ub.tum.de/s1_africa.tar.gz", 
    "data/s1_africa.tar.gz",
    reporthook=show_progress
)
print("\nSAR Download complete!")

print("\nDownloading Sentinel-2 Optical for Africa...")
urllib.request.urlretrieve(
    "ftp://m1639953:m1639953@dataserv.ub.tum.de/s2_africa.tar.gz", 
    "data/s2_africa.tar.gz",
    reporthook=show_progress
)
print("\nOptical Download complete! You are ready to extract.")