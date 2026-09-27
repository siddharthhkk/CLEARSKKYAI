import argparse
import math
import os
import sys

import numpy as np
import rasterio
import rasterio.warp
import torch
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR


BANDS = ["BAND2", "BAND3", "BAND4"]


def find_band_path(row_folder, names, band):
    for name in names:
        if name.upper() in (f"{band}.TIF", f"{band}.TIFF"):
            return os.path.join(row_folder, name)

    for name in names:
        if band.lower() in name.lower():
            return os.path.join(row_folder, name)

    raise FileNotFoundError(f"{band} not found in {row_folder}")


def load_inventory(path):
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_scene(rows, scene_id):
    for row in rows:
        if row["scene_id"] == scene_id:
            return row
    raise SystemExit(f"FAIL: scene not found in inventory: {scene_id}")


def common_bounds(a, b):
    l = max(float(a["left"]), float(b["left"]))
    bot = max(float(a["bottom"]), float(b["bottom"]))
    r = min(float(a["right"]), float(b["right"]))
    top = min(float(a["top"]), float(b["top"]))

    if r <= l or top <= bot:
        raise SystemExit("FAIL: scenes do not overlap.")

    return l, bot, r, top


def load_common_grid(scene_a, scene_b, bounds, dn_max):
    """
    Build one common 5 m grid from scene B's native grid, restricted to
    the geographic overlap, then warp scene A onto exactly that grid.
    """
    b_paths = scene_b["tif_files"].split(";")
    b0 = find_band_path(scene_b["folder"], b_paths, BANDS[0])

    with rasterio.open(b0) as ref:
        win = ref.window(*bounds)
        win = win.round_offsets().round_lengths()

        h = int(win.height)
        w = int(win.width)
        if h <= 0 or w <= 0:
            raise SystemExit("FAIL: overlap window is empty.")

        transform = ref.transform * Affine.translation(
            int(win.col_off), int(win.row_off)
        )

        profile = ref.profile.copy()
        profile.update(
            width=w,
            height=h,
            count=3,
            dtype="uint16",
            transform=transform,
            compress="deflate",
        )

        clear = np.zeros((3, h, w), dtype=np.float32)

        for i, band in enumerate(BANDS):
            p = find_band_path(scene_b["folder"], b_paths, band)
            with rasterio.open(p) as src:
                clear[i] = src.read(
                    1,
                    window=win,
                    out_shape=(h, w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

        cloudy = np.zeros((3, h, w), dtype=np.float32)

        a_paths = scene_a["tif_files"].split(";")
        for i, band in enumerate(BANDS):
            p = find_band_path(scene_a["folder"], a_paths, band)
            with rasterio.open(p) as src:
                dst = np.zeros((h, w), dtype=np.float32)

                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=0,
                    dst_nodata=0,
                )

                cloudy[i] = dst

    cloudy = np.clip(cloudy, 0, dn_max)
    clear = np.clip(clear, 0, dn_max)

    return cloudy, clear, profile, transform


def weight_map(h, w):
    y = np.clip(
        np.sin(np.pi * np.linspace(0, 1, h, dtype=np.float32)),
        0,
        1,
    )
    x = np.clip(
        np.sin(np.pi * np.linspace(0, 1, w, dtype=np.float32)),
        0,
        1,
    )
    return (
        (0.15 + 0.85 * np.sqrt(y))[:, None]
        * (0.15 + 0.85 * np.sqrt(x))[None, :]
    ).astype(np.float32)


def starts(n, tile, stride):
    if n <= tile:
        return [0]

    out = list(range(0, n - tile + 1, stride))
    last = n - tile
    if out[-1] != last:
        out.append(last)
    return out


def infer(model, x, device, dn_max, tile, overlap):
    _, h, w = x.shape
    stride = tile - overlap

    rr = starts(h, tile, stride)
    cc = starts(w, tile, stride)

    acc = np.zeros_like(x, dtype=np.float32)
    ws = np.zeros((h, w), dtype=np.float32)
    wt_cache = {}

    total = len(rr) * len(cc)
    done = 0

    for row in rr:
        for col in cc:
            row2 = min(row + tile, h)
            col2 = min(col + tile, w)
            hh = row2 - row
            ww = col2 - col

            patch = x[:, row:row2, col:col2]

            if hh != tile or ww != tile:
                patch = np.pad(
                    patch,
                    ((0, 0), (0, tile - hh), (0, tile - ww)),
                    mode="edge",
                )

            xt = normalize_liss4(patch, dn_max)[None].to(device)

            with torch.inference_mode():
                yp = torch.clamp(model(xt), 0, 1)

            pred = (
                yp[0]
                .detach()
                .cpu()
                .numpy()[:, :hh, :ww]
            )

            key = (hh, ww)
            if key not in wt_cache:
                wt_cache[key] = weight_map(hh, ww)

            wt = wt_cache[key]

            acc[:, row:row2, col:col2] += pred * wt[None]
            ws[row:row2, col:col2] += wt

            done += 1
            if done == 1 or done % 100 == 0 or done == total:
                print(
                    f"  inference {done}/{total} tiles "
                    f"({100.0 * done / total:.1f}%)",
                    flush=True,
                )

    if not np.all(ws > 0):
        raise RuntimeError("Inference blending left uncovered pixels.")

    return acc / ws[None]


def metrics(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    rmse = float(math.sqrt(mse))
    psnr = (
        float(20 * math.log10(1.0 / math.sqrt(mse)))
        if mse > 0
        else float("inf")
    )
    return mae, rmse, psnr


def ssim_band(pred, target, window=11):
    p = pred.astype(np.float64)
    t = target.astype(np.float64)

    pad = window // 2

    def box(z):
        zp = np.pad(z, ((pad, pad), (pad, pad)), mode="reflect")
        cs = np.pad(zp, ((1, 0), (1, 0)), mode="constant")
        cs = cs.cumsum(0).cumsum(1)
        h, w = z.shape
        return (
            cs[window:window + h, window:window + w]
            - cs[:h, window:window + w]
            - cs[window:window + h, :w]
            + cs[:h, :w]
        ) / float(window * window)

    mp = box(p)
    mt = box(t)
    vp = np.maximum(box(p * p) - mp * mp, 0)
    vt = np.maximum(box(t * t) - mt * mt, 0)
    cov = box(p * t) - mp * mt

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    num = (2 * mp * mt + c1) * (2 * cov + c2)
    den = (mp * mp + mt * mt + c1) * (vp + vt + c2)

    return float(np.mean(num / np.maximum(den, 1e-12)))


def ssim(pred, target):
    return float(
        np.mean(
            [
                ssim_band(pred[i], target[i])
                for i in range(pred.shape[0])
            ]
        )
    )


def sam(pred, target):
    p = pred.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64)
    t = target.transpose(1, 2, 0).reshape(-1, 3).astype(np.float64)

    np1 = np.linalg.norm(p, axis=1)
    np2 = np.linalg.norm(t, axis=1)

    valid = (np1 > 1e-8) & (np2 > 1e-8)
    cos = np.sum(p * t, axis=1) / np.maximum(np1 * np2, 1e-12)
    return float(np.degrees(np.mean(np.arccos(np.clip(cos[valid], -1, 1)))))


def print_metrics(label, x, y):
    mae, rmse, psnr = metrics(x, y)
    s = ssim(x, y)
    a = sam(x, y)

    print(
        f"{label:<8} MAE={mae:.6f} "
        f"RMSE={rmse:.6f} "
        f"PSNR={psnr:.3f} dB "
        f"SSIM={s:.5f} "
        f"SAM={a:.4f}°"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": s,
        "sam": a,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run V3R on the geographic overlap of two real Bhoonidhi "
            "LISS-IV scenes and compare against the second scene."
        )
    )
    ap.add_argument("cloudy_scene")
    ap.add_argument("clear_scene")
    ap.add_argument(
        "--csv",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    ap.add_argument(
        "--checkpoint",
        default="weights/liss4_dsen2cr_synthetic_v3r.pth",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument(
        "--output",
        default="data/eval/bhoonidhi_v3r_prediction.tif",
    )
    ap.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save aligned cloudy/clear/pred arrays as compressed NPZ.",
    )
    args = ap.parse_args()

    if not 0 <= args.overlap < args.tile:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile.")

    rows = load_inventory(os.path.abspath(args.csv))
    a = find_scene(rows, args.cloudy_scene)
    b = find_scene(rows, args.clear_scene)

    bounds = common_bounds(a, b)

    print("=== BHOONIDHI REAL-PAIR V3R EVALUATION ===")
    print(f"Cloudy candidate : {a['scene_id']} ({a['date']})")
    print(f"Clear candidate  : {b['scene_id']} ({b['date']})")
    print(f"Path/row         : {a['path']}/{a['row']}")
    print(
        f"Overlap bounds   : "
        f"{bounds[0]:.2f}, {bounds[1]:.2f}, "
        f"{bounds[2]:.2f}, {bounds[3]:.2f}"
    )
    print()

    cloudy_dn, clear_dn, profile, transform = load_common_grid(
        a,
        b,
        bounds,
        args.dn_max,
    )

    h, w = cloudy_dn.shape[1:]
    print(f"Common grid      : {w} x {h} pixels")
    print(f"Pixel size       : {abs(transform.a):.3f} x {abs(transform.e):.3f}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.abspath(args.checkpoint)
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model = LISS4DSen2CR(
        features=int(ckpt.get("features", 256)),
        blocks=int(ckpt.get("blocks", 16)),
        res_scale=0.1,
        use_sar=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Device           : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint       : {ckpt_path}")
    print()

    pred_norm = infer(
        model,
        cloudy_dn,
        device,
        args.dn_max,
        args.tile,
        args.overlap,
    )

    pred_dn = np.clip(np.rint(pred_norm * args.dn_max), 0, args.dn_max)
    pred_norm = pred_dn.astype(np.float32) / args.dn_max

    cloudy_norm = cloudy_dn.astype(np.float32) / args.dn_max
    clear_norm = clear_dn.astype(np.float32) / args.dn_max

    print()
    print("=== METRICS ON COMMON OVERLAP ===")
    base = print_metrics("Cloudy", cloudy_norm, clear_norm)
    out = print_metrics("V3R", pred_norm, clear_norm)

    print()
    print("=== CHANGE VS CLOUDY BASELINE ===")
    print(
        f"MAE  change : {out['mae'] - base['mae']:+.6f} "
        f"(negative = lower error)"
    )
    print(
        f"RMSE change : {out['rmse'] - base['rmse']:+.6f} "
        f"(negative = lower error)"
    )
    print(
        f"PSNR gain   : {out['psnr'] - base['psnr']:+.3f} dB"
    )
    print(
        f"SSIM change : {out['ssim'] - base['ssim']:+.5f}"
    )
    print(
        f"SAM change  : {out['sam'] - base['sam']:+.4f}° "
        f"(negative = lower angle)"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    profile.update(
        dtype="uint16",
        count=3,
        width=w,
        height=h,
        transform=transform,
    )

    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(pred_dn.astype(np.uint16))

    print()
    print(f"Prediction TIFF : {os.path.abspath(args.output)}")

    if args.save_npz:
        npz_path = os.path.splitext(os.path.abspath(args.output))[0] + ".npz"
        np.savez_compressed(
            npz_path,
            cloudy=cloudy_dn.astype(np.uint16),
            clear=clear_dn.astype(np.uint16),
            pred=pred_dn.astype(np.uint16),
            bounds=np.asarray(bounds, dtype=np.float64),
        )
        print(f"Aligned NPZ     : {npz_path}")

    print()
    print(
        "Caveat: the second acquisition is a temporal reference, not a "
        "simultaneous cloud-free ground truth. Metrics include land-cover, "
        "illumination, registration, and seasonal differences."
    )


if __name__ == "__main__":
    main()
