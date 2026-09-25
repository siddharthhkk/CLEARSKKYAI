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

from dsen2cr import DSen2CR
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


def dsen2cr_sar_preprocess(arr):
    """Official DSen2-CR normalization: SAR -> [0, 2]."""
    x = np.asarray(arr, dtype=np.float32).copy()

    mins = np.array([-25.0, -32.5], dtype=np.float32)
    maxs = np.array([0.0, 0.0], dtype=np.float32)

    for i in range(2):
        x[i] = np.clip(x[i], mins[i], maxs[i])
        x[i] = 2.0 * (x[i] - mins[i]) / (maxs[i] - mins[i])

    return torch.from_numpy(x).float()


def dsen2cr_optical_preprocess(arr):
    """Official DSen2-CR normalization: optical -> [0, 5]."""
    x = np.asarray(arr, dtype=np.float32)
    x = np.clip(x, 0.0, 10000.0)
    return torch.from_numpy(x / 2000.0).float()


def to_unit(x, scale):
    return torch.clamp(x / scale, 0.0, 1.0)


def save_gray(arr, path):
    x = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(x, (2, 98))

    if hi > lo:
        x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    else:
        x = np.zeros_like(x)

    Image.fromarray((x * 255.0).astype(np.uint8)).save(path)


def save_rgb(arr, path):
    x = np.clip(np.transpose(arr, (1, 2, 0)), 0.0, 1.0)
    Image.fromarray((x * 255.0).astype(np.uint8)).save(path)


def save_shared_rgb(out_rgb, gt_rgb, out_path, gt_path):
    out_rgb = out_rgb.astype(np.float32)
    gt_rgb = gt_rgb.astype(np.float32)

    a = np.empty_like(out_rgb)
    b = np.empty_like(gt_rgb)

    for i in range(3):
        vals = np.concatenate([out_rgb[i].ravel(), gt_rgb[i].ravel()])
        lo, hi = np.percentile(vals, (2, 98))

        if hi > lo:
            a[i] = np.clip((out_rgb[i] - lo) / (hi - lo), 0.0, 1.0)
            b[i] = np.clip((gt_rgb[i] - lo) / (hi - lo), 0.0, 1.0)
        else:
            a[i] = 0.0
            b[i] = 0.0

    save_rgb(a, out_path)
    save_rgb(b, gt_path)


def print_stats(label, arr, names):
    print(f"\n{label}")
    for i, ch in enumerate(names):
        x = arr[i].astype(np.float32)
        print(
            f"  {ch}: mean={x.mean():.6f} "
            f"std={x.std():.6f} "
            f"min={x.min():.6f} "
            f"max={x.max():.6f} "
            f"p2={np.percentile(x, 2):.6f} "
            f"p98={np.percentile(x, 98):.6f}"
        )


def calc_psnr(pred, target, peak):
    mse = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak / math.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose ClearSky-AI or DSen2-CR output on a demo tile."
    )
    parser.add_argument(
        "--tile",
        required=True,
        help="Exact SAR filename in demo_samples/sar/",
    )
    parser.add_argument(
        "--model",
        choices=["clearsky", "dsen2cr"],
        default="clearsky",
        help="Model to evaluate. Default: clearsky",
    )
    parser.add_argument(
        "--out",
        default="temp/diagnostics",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Synthetic-cloud seed. Defaults to the tile-derived seed.",
    )
    args = parser.parse_args()

    pairs = find_demo_pairs()
    match = next(
        (
            pair
            for pair in pairs
            if os.path.basename(pair[0]) == args.tile
        ),
        None,
    )

    if match is None:
        print("❌ Tile not found.")
        print("Available tiles:")
        for sar_path, _ in pairs:
            print(f"  {os.path.basename(sar_path)}")
        raise SystemExit(1)

    sar_path, opt_path = match

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    weights_path = (
        os.path.join(PROJECT_ROOT, "weights", "best_model.pth")
        if args.model == "clearsky"
        else os.path.join(PROJECT_ROOT, "weights", "dsen2cr_sar_carl.pth")
    )

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    if args.model == "clearsky":
        model = ClearSkyUNet().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        model = DSen2CR().to(device)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model = model.to(device=device, dtype=torch.float32)

    model.eval()

    with rasterio.open(sar_path) as src:
        sar_raw = src.read().astype(np.float32)

    with rasterio.open(opt_path) as src:
        if args.model == "clearsky":
            if src.count < 8:
                raise ValueError(f"Expected at least 8 S2 bands, found {src.count}")
            opt_raw = src.read([4, 3, 2, 8]).astype(np.float32)
        else:
            if src.count < 13:
                raise ValueError(f"DSen2-CR requires 13 S2 bands, found {src.count}")
            opt_raw = src.read(list(range(1, 14))).astype(np.float32)

    if args.model == "clearsky":
        sar_tensor = normalize_sar(sar_raw)
        target_tensor = normalize_optical(opt_raw)
        rgb_idx = [0, 1, 2]
        scale = 1.0
        peak = 2.0
        names = ["R", "G", "B"]
    else:
        sar_tensor = dsen2cr_sar_preprocess(sar_raw)
        target_tensor = dsen2cr_optical_preprocess(opt_raw)
        rgb_idx = [3, 2, 1]
        scale = 5.0
        peak = 1.0
        names = [f"B{i:02d}" for i in range(1, 14)]

    seed = (
        args.seed
        if args.seed is not None
        else sum(ord(ch) for ch in os.path.basename(sar_path)) % (2**31 - 1)
    )

    cloudy_tensor, cloud_mask = make_synthetic_cloud(target_tensor, seed)

    with torch.no_grad():
        if args.model == "clearsky":
            pred = model(
                sar_tensor.unsqueeze(0).to(device),
                cloudy_tensor.unsqueeze(0).to(device),
            ).squeeze(0).cpu()
        else:
            inp = torch.cat(
                [cloudy_tensor, sar_tensor],
                dim=0,
            ).unsqueeze(0).to(device=device, dtype=torch.float32)

            pred = model(inp).squeeze(0).cpu()

    target = target_tensor.cpu().numpy()
    output = pred.numpy()
    cloud_px = cloud_mask.squeeze(0).cpu().numpy().astype(bool)
    visible_px = ~cloud_px

    print(f"\n📂 Tile: {os.path.basename(sar_path)}")
    print(f"🧠 Model: {args.model}")
    print(f"☁️ Cloud seed: {seed}")
    print(f"☁️ Cloud coverage: {cloud_mask.mean().item() * 100.0:.2f}%")

    print_stats("GROUND TRUTH", target, names)
    print_stats("MODEL OUTPUT", output, names)

    rgb_out = output[rgb_idx] / scale
    rgb_tgt = target[rgb_idx] / scale

    rgb_out = np.clip(rgb_out, 0.0, 1.0)
    rgb_tgt = np.clip(rgb_tgt, 0.0, 1.0)

    err = np.abs(rgb_out - rgb_tgt)

    print("\nRGB error")
    print(f"  Overall RGB MAE: {err.mean():.6f}")
    print(f"  R MAE: {err[0].mean():.6f}")
    print(f"  G MAE: {err[1].mean():.6f}")
    print(f"  B MAE: {err[2].mean():.6f}")

    print(f"  Overall RGB PSNR: {calc_psnr(rgb_out, rgb_tgt, peak):.2f} dB")

    if np.any(cloud_px):
        ce = err[:, cloud_px]
        print(f"  Cloud RGB MAE: {ce.mean():.6f}")
        print(
            f"  Cloud RGB PSNR: "
            f"{calc_psnr(rgb_out[:, cloud_px], rgb_tgt[:, cloud_px], peak):.2f} dB"
        )

    if np.any(visible_px):
        print(f"  Visible RGB MAE: {err[:, visible_px].mean():.6f}")

    out_dir = os.path.join(
        args.out,
        f"{os.path.splitext(args.tile)[0]}_{args.model}",
    )
    os.makedirs(out_dir, exist_ok=True)

    for i, ch in enumerate(names[:3] if args.model == "clearsky" else ["R", "G", "B"]):
        save_gray(
            target[rgb_idx[i]],
            os.path.join(out_dir, f"gt_{ch}.png"),
        )
        save_gray(
            output[rgb_idx[i]],
            os.path.join(out_dir, f"output_{ch}.png"),
        )
        save_gray(
            err[i],
            os.path.join(out_dir, f"diff_{ch}.png"),
        )

    save_gray(
        cloud_mask.squeeze(0).cpu().numpy(),
        os.path.join(out_dir, "cloud_mask.png"),
    )

    cloudy_rgb = np.clip(
        cloudy_tensor.cpu().numpy()[rgb_idx] / scale,
        0.0,
        1.0,
    )
    save_rgb(
        cloudy_rgb,
        os.path.join(out_dir, "cloudy_rgb.png"),
    )

    save_shared_rgb(
        rgb_out,
        rgb_tgt,
        os.path.join(out_dir, "output_rgb_shared.png"),
        os.path.join(out_dir, "gt_rgb_shared.png"),
    )

    print(f"\n✅ Saved diagnostic files to: {os.path.abspath(out_dir)}")
    print("   gt_R.png / output_R.png / diff_R.png")
    print("   gt_G.png / output_G.png / diff_G.png")
    print("   gt_B.png / output_B.png / diff_B.png")
    print("   output_rgb_shared.png / gt_rgb_shared.png")
    print("   cloud_mask.png / cloudy_rgb.png")


if __name__ == "__main__":
    main()
