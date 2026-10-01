import argparse
import math
import os

import numpy as np
import rasterio


def load(path, dn_max):
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)

    if x.shape[0] != 3:
        raise ValueError(f"{path}: expected 3 bands [G,R,NIR], got {x.shape}")

    return np.clip(x, 0.0, dn_max) / dn_max


def basic_metrics(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    rmse = float(math.sqrt(mse))
    psnr = (
        float(20.0 * math.log10(1.0 / math.sqrt(mse)))
        if mse > 0
        else float("inf")
    )
    return mae, rmse, psnr


def box_mean(x, k):
    # Efficient 2D box filter using cumulative sums.
    p = k // 2
    xp = np.pad(x, ((p, p), (p, p)), mode="reflect")
    cs = np.pad(xp, ((1, 0), (1, 0)), mode="constant")
    cs = cs.cumsum(axis=0).cumsum(axis=1)

    h, w = x.shape
    y = (
        cs[k:k + h, k:k + w]
        - cs[:h, k:k + w]
        - cs[k:k + h, :w]
        + cs[:h, :w]
    )
    return y / float(k * k)


def ssim_band(pred, target, window=11, data_range=1.0):
    p = pred.astype(np.float64)
    t = target.astype(np.float64)

    mu_p = box_mean(p, window)
    mu_t = box_mean(t, window)

    mu_p2 = box_mean(p * p, window)
    mu_t2 = box_mean(t * t, window)
    mu_pt = box_mean(p * t, window)

    var_p = np.maximum(mu_p2 - mu_p * mu_p, 0.0)
    var_t = np.maximum(mu_t2 - mu_t * mu_t, 0.0)
    cov = mu_pt - mu_p * mu_t

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_p * mu_t + c1) * (2.0 * cov + c2)
    den = (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)

    return float(np.mean(num / np.maximum(den, 1e-12)))


def ssim(pred, target, window=11):
    return float(
        np.mean(
            [ssim_band(pred[i], target[i], window=window) for i in range(pred.shape[0])]
        )
    )


def sam_degrees(pred, target):
    p = pred.astype(np.float64).transpose(1, 2, 0).reshape(-1, 3)
    t = target.astype(np.float64).transpose(1, 2, 0).reshape(-1, 3)

    dot = np.sum(p * t, axis=1)
    np_p = np.linalg.norm(p, axis=1)
    np_t = np.linalg.norm(t, axis=1)

    denom = np.maximum(np_p * np_t, 1e-12)
    cos = np.clip(dot / denom, -1.0, 1.0)

    valid = (np_p > 1e-8) & (np_t > 1e-8)
    angles = np.arccos(cos[valid])

    return float(np.degrees(np.mean(angles))) if angles.size else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate Guwahati LISS-IV cloud removal with MAE, RMSE, PSNR, "
            "SSIM, and spectral angle mapper (SAM)."
        )
    )
    ap.add_argument("--cloudy", default="data/raw/cloudy/guwahati_cloudy_test.tif")
    ap.add_argument("--clear", default="data/raw/clear/guwahati_clear.tif")
    ap.add_argument("--v2", default="data/eval/guwahati_liss4_dsen2cr_v2.tif")
    ap.add_argument("--v3r", default="data/eval/guwahati_liss4_dsen2cr_v3r.tif")
    ap.add_argument(
        "--baseline-label",
        default="V2",
        help="Display label for the --v2 raster (default: V2).",
    )
    ap.add_argument(
        "--candidate-label",
        default="V3R",
        help="Display label for the --v3r raster (default: V3R).",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument("--ssim-window", type=int, default=11)
    args = ap.parse_args()

    if args.ssim_window < 3 or args.ssim_window % 2 == 0:
        raise ValueError("--ssim-window must be an odd integer >= 3.")

    target = load(args.clear, args.dn_max)
    cloudy = load(args.cloudy, args.dn_max)
    v2 = load(args.v2, args.dn_max)
    v3r = load(args.v3r, args.dn_max)

    for name, x in [("Cloudy", cloudy), ("V2", v2), ("V3R", v3r)]:
        if x.shape != target.shape:
            raise ValueError(f"{name} shape {x.shape} != target shape {target.shape}")

    print("=== GUWAHATI LISS-IV MULTI-METRIC EVALUATION ===")
    print("Bands: [G, R, NIR]")
    print("PSNR/MAE/RMSE use normalized DN range [0,1].")
    print("SSIM is the mean of per-band windowed SSIM.")
    print("SAM is mean spectral angle in degrees; lower is better.")
    print()

    for name, pred in [
        ("Cloudy", cloudy),
        (args.baseline_label, v2),
        (args.candidate_label, v3r),
    ]:
        mae, rmse, psnr = basic_metrics(pred, target)
        score_ssim = ssim(pred, target, window=args.ssim_window)
        score_sam = sam_degrees(pred, target)

        print(
            f"{name:<6} MAE={mae:.6f} "
            f"RMSE={rmse:.6f} "
            f"PSNR={psnr:.3f} dB "
            f"SSIM={score_ssim:.5f} "
            f"SAM={score_sam:.4f}°"
        )

    print()
    print(
        "Interpretation: Guwahati cloudy/clear is a temporal pair, not a "
        "simultaneous cloud-mask ground truth. Metrics therefore also reflect "
        "illumination, registration, land-surface change, and other acquisition differences."
    )


if __name__ == "__main__":
    main()
