import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from dsen2cr import DSen2CR


def calc_psnr(pred, target, peak=1.0):
    mse = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak / math.sqrt(mse))


def save_rgb(arr, path):
    x = np.clip(np.transpose(arr, (1, 2, 0)), 0.0, 1.0)
    Image.fromarray((x * 255.0).astype(np.uint8)).save(path)


def save_gray(arr, path):
    x = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(x, (2, 98))
    if hi > lo:
        x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    else:
        x = np.zeros_like(x)
    Image.fromarray((x * 255.0).astype(np.uint8)).save(path)


def main():
    p = argparse.ArgumentParser(
        description="Run pretrained DSen2-CR on a real SEN12MS-CR triplet."
    )
    p.add_argument("--sample", required=True, help="Path to sample_XXX.npz")
    p.add_argument("--out", default="temp/real_dsen2cr")
    args = p.parse_args()

    sample_path = os.path.abspath(args.sample)
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(sample_path)

    d = np.load(sample_path, allow_pickle=True)

    sar = d["sar"].astype(np.float32)
    cloudy = d["cloudy"].astype(np.float32)
    target = d["target"].astype(np.float32)

    if sar.shape != (2, 256, 256):
        raise ValueError(f"Expected SAR [2,256,256], got {sar.shape}")
    if cloudy.shape != (13, 256, 256):
        raise ValueError(f"Expected cloudy S2 [13,256,256], got {cloudy.shape}")
    if target.shape != (13, 256, 256):
        raise ValueError(f"Expected target S2 [13,256,256], got {target.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    weights = os.path.join(PROJECT_ROOT, "weights", "dsen2cr_sar_carl.pth")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Missing weights: {weights}")

    model = DSen2CR().to(device=device, dtype=torch.float32)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # SEN12MS-CR optical values are stored as reflectance-like values [0,10000].
    # DSen2-CR expects [0,5]. SAR is converted to [0,2].
    cloudy_in = np.clip(cloudy, 0.0, 10000.0) / 2000.0
    target_in = np.clip(target, 0.0, 10000.0) / 2000.0

    sar_in = sar.copy()
    sar_in[0] = 2.0 * (
        np.clip(sar_in[0], -25.0, 0.0) + 25.0
    ) / 25.0
    sar_in[1] = 2.0 * (
        np.clip(sar_in[1], -35.0, 0.0) + 35.0
    ) / 35.0

    x = torch.from_numpy(np.concatenate([cloudy_in, sar_in], axis=0))
    x = x.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        out = model(x).squeeze(0).cpu().numpy()

    # RGB = B04, B03, B02 = indices 3,2,1.
    ridx = [3, 2, 1]
    out_rgb = np.clip(out[ridx] / 5.0, 0.0, 1.0)
    tgt_rgb = np.clip(target_in[ridx] / 5.0, 0.0, 1.0)
    in_rgb = np.clip(cloudy_in[ridx] / 5.0, 0.0, 1.0)

    err = np.abs(out_rgb - tgt_rgb)

    print(f"\n📂 Sample: {os.path.basename(sample_path)}")
    for key in ("season", "scene", "patch", "source"):
        if key in d:
            print(f"   {key}: {d[key].item() if d[key].shape == () else d[key]}")

    print("\nRGB metrics on REAL cloudy input")
    print(f"  RGB MAE: {err.mean():.6f}")
    print(f"  R MAE: {err[0].mean():.6f}")
    print(f"  G MAE: {err[1].mean():.6f}")
    print(f"  B MAE: {err[2].mean():.6f}")
    print(f"  RGB PSNR: {calc_psnr(out_rgb, tgt_rgb):.2f} dB")

    # This is a real-cloud sample, but this mirror does not provide a pixel mask.
    # Do not label a difference-derived heuristic as a true cloud mask.
    print("  Cloud-region metric: not reported (no native cloud mask in mirror)")

    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(sample_path))[0])
    os.makedirs(out_dir, exist_ok=True)

    save_rgb(in_rgb, os.path.join(out_dir, "real_cloudy_rgb.png"))
    save_rgb(out_rgb, os.path.join(out_dir, "dsen2cr_output_rgb.png"))
    save_rgb(tgt_rgb, os.path.join(out_dir, "ground_truth_rgb.png"))

    save_gray(
        np.mean(err, axis=0),
        os.path.join(out_dir, "rgb_error.png"),
    )

    print(f"\n✅ Saved outputs to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
