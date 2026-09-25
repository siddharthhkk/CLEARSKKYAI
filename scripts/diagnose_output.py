import argparse
import glob
import math
import os
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from models import ClearSkyUNet
from preprocess import normalize_optical, normalize_sar


def find_demo_pairs():
    sar_dir = os.path.join(PROJECT_ROOT, "demo_samples", "sar")
    opt_dir = os.path.join(PROJECT_ROOT, "demo_samples", "optical")

    sar_files = sorted(
        os.path.join(sar_dir, f)
        for f in os.listdir(sar_dir)
        if f.lower().startswith("s1_") and f.lower().endswith(".tif")
    )

    pairs = []
    for sar_path in sar_files:
        opt_name = os.path.basename(sar_path).replace("s1_", "s2_", 1)
        opt_path = os.path.join(opt_dir, opt_name)
        if os.path.isfile(opt_path):
            pairs.append((sar_path, opt_path))

    return pairs


def make_synthetic_cloud(opt_tensor, seed):
    _, h, w = opt_tensor.shape
    nh = max(1, h // 16)
    nw = max(1, w // 16)

    g = torch.Generator()
    g.manual_seed(seed)

    noise = torch.rand(1, 1, nh, nw, generator=g)

    mask = F.interpolate(
        noise,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    mask = (mask > 0.65).float()
    cloudy = opt_tensor * (1.0 - mask) + mask * 1.0

    return cloudy, mask


def to_unit(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def save_gray(arr, path):
    x = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(x, (2, 98))

    if hi > lo:
        x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    else:
        x = np.zeros_like(x)

    Image.fromarray((x * 255.0).astype(np.uint8), mode="L").save(path)


def save_shared_channel(a, b, path_a, path_b):
    for i, ch in enumerate(("R", "G", "B")):
        vals = np.concatenate(
            [a[i].ravel().astype(np.float32), b[i].ravel().astype(np.float32)]
        )
        lo, hi = np.percentile(vals, (2, 98))

        if hi > lo:
            aa = np.clip((a[i] - lo) / (hi - lo), 0.0, 1.0)
            bb = np.clip((b[i] - lo) / (hi - lo), 0.0, 1.0)
        else:
            aa = np.zeros_like(a[i])
            bb = np.zeros_like(b[i])

        Image.fromarray((aa * 255.0).astype(np.uint8), mode="L").save(
            os.path.join(path_a, f"R_dummy_{ch}.png")
        )
        Image.fromarray((bb * 255.0).astype(np.uint8), mode="L").save(
            os.path.join(path_b, f"R_dummy_{ch}.png")
        )


def print_rgb_stats(label, arr):
    print(f"\n{label}")
    for i, ch in enumerate(("R", "G", "B")):
        x = arr[i].astype(np.float32)
        print(
            f"  {ch}: mean={x.mean():.6f} "
            f"std={x.std():.6f} "
            f"min={x.min():.6f} "
            f"max={x.max():.6f} "
            f"p2={np.percentile(x, 2):.6f} "
            f"p98={np.percentile(x, 98):.6f}"
        )


def save_shared_rgb_pair(out_rgb, gt_rgb, out_dir):
    out_path = os.path.join(out_dir, "output_rgb_shared.png")
    gt_path = os.path.join(out_dir, "gt_rgb_shared.png")

    a = out_rgb.astype(np.float32)
    b = gt_rgb.astype(np.float32)
    ao = np.empty_like(a)
    bo = np.empty_like(b)

    for i in range(3):
        vals = np.concatenate([a[i].ravel(), b[i].ravel()])
        lo, hi = np.percentile(vals, (2, 98))

        if hi > lo:
            ao[i] = np.clip((a[i] - lo) / (hi - lo), 0.0, 1.0)
            bo[i] = np.clip((b[i] - lo) / (hi - lo), 0.0, 1.0)
        else:
            ao[i] = 0.0
            bo[i] = 0.0

    Image.fromarray((np.transpose(ao, (1, 2, 0)) * 255).astype(np.uint8)).save(
        out_path
    )
    Image.fromarray((np.transpose(bo, (1, 2, 0)) * 255).astype(np.uint8)).save(
        gt_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose ClearSky-AI RGB output on a demo tile."
    )
    parser.add_argument(
        "--tile",
        required=True,
        help="Exact SAR filename in demo_samples/sar/",
    )
    parser.add_argument(
        "--out",
        default="temp/diagnostics",
        help="Output directory for diagnostic images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Synthetic-cloud seed. Defaults to the same tile-derived seed used by the app.",
    )
    args = parser.parse_args()

    pairs = find_demo_pairs()
    match = None

    for sar_path, opt_path in pairs:
        if os.path.basename(sar_path) == args.tile:
            match = (sar_path, opt_path)
            break

    if match is None:
        print("❌ Tile not found.")
        print("Available tiles:")
        for sar_path, _ in pairs:
            print(f"  {os.path.basename(sar_path)}")
        raise SystemExit(1)

    sar_path, opt_path = match

    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    weights_path = os.path.join(PROJECT_ROOT, "weights", "best_model.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    model = ClearSkyUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with rasterio.open(sar_path) as src:
        sar_raw = src.read().astype(np.float32)

    with rasterio.open(opt_path) as src:
        if src.count < 8:
            raise ValueError(f"Expected at least 8 S2 bands, found {src.count}")
        opt_raw = src.read([4, 3, 2, 8]).astype(np.float32)

    sar_tensor = normalize_sar(sar_raw)
    target_tensor = normalize_optical(opt_raw)

    seed = (
        args.seed
        if args.seed is not None
        else sum(ord(ch) for ch in os.path.basename(sar_path)) % (2**31 - 1)
    )

    cloudy_tensor, cloud_mask = make_synthetic_cloud(target_tensor, seed)

    with torch.no_grad():
        output_tensor = model(
            sar_tensor.unsqueeze(0).to(device),
            cloudy_tensor.unsqueeze(0).to(device),
        ).squeeze(0).cpu()

    target = target_tensor.cpu().numpy()
    output = output_tensor.numpy()

    print(f"\n📂 Tile: {os.path.basename(sar_path)}")
    print(f"☁️ Cloud seed: {seed}")
    print(f"☁️ Cloud coverage: {cloud_mask.mean().item() * 100.0:.2f}%")

    print_rgb_stats("GROUND TRUTH RGB — normalized model space", target)
    print_rgb_stats("MODEL OUTPUT RGB — normalized model space", output)

    err = np.abs(output[:3] - target[:3])

    print("\nRGB error")
    print(f"  Overall RGB MAE: {err.mean():.6f}")
    print(f"  R MAE: {err[0].mean():.6f}")
    print(f"  G MAE: {err[1].mean():.6f}")
    print(f"  B MAE: {err[2].mean():.6f}")

    mse = np.mean((output - target) ** 2)
    psnr = float("inf") if mse <= 0 else 20.0 * math.log10(2.0 / math.sqrt(mse))
    print(f"  Overall 4-channel PSNR: {psnr:.2f} dB")

    cloud_px = cloud_mask.squeeze(0).cpu().numpy().astype(bool)
    visible_px = ~cloud_px

    if np.any(cloud_px):
        ce = err[:, cloud_px]
        cmse = np.mean((output[:3, cloud_px] - target[:3, cloud_px]) ** 2)
        cpsnr = float("inf") if cmse <= 0 else 20.0 * math.log10(2.0 / math.sqrt(cmse))
        print(f"  Cloud RGB MAE: {ce.mean():.6f}")
        print(f"  Cloud RGB PSNR: {cpsnr:.2f} dB")

    if np.any(visible_px):
        print(f"  Visible RGB MAE: {err[:, visible_px].mean():.6f}")

    out_dir = os.path.join(args.out, os.path.splitext(args.tile)[0])
    os.makedirs(out_dir, exist_ok=True)

    for i, ch in enumerate(("R", "G", "B")):
        save_gray(target[i], os.path.join(out_dir, f"gt_{ch}.png"))
        save_gray(output[i], os.path.join(out_dir, f"output_{ch}.png"))
        save_gray(err[i], os.path.join(out_dir, f"diff_{ch}.png"))

    save_gray(
        cloud_mask.squeeze(0).cpu().numpy(),
        os.path.join(out_dir, "cloud_mask.png"),
    )

    cloudy = to_unit(cloudy_tensor).cpu().numpy()
    Image.fromarray(
        (np.transpose(cloudy[:3], (1, 2, 0)) * 255).astype(np.uint8)
    ).save(os.path.join(out_dir, "cloudy_rgb.png"))

    save_shared_rgb_pair(output[:3], target[:3], out_dir)

    print(f"\n✅ Saved diagnostic files to: {os.path.abspath(out_dir)}")
    print("   gt_R.png / output_R.png / diff_R.png")
    print("   gt_G.png / output_G.png / diff_G.png")
    print("   gt_B.png / output_B.png / diff_B.png")
    print("   output_rgb_shared.png / gt_rgb_shared.png")
    print("   cloud_mask.png / cloudy_rgb.png")


if __name__ == "__main__":
    main()
