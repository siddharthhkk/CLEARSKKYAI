import argparse
import math
import os

import numpy as np
import rasterio


def metric(pred, target, mask):
    if not np.any(mask):
        return float("nan"), float("nan")

    d = pred[:, mask] - target[:, mask]
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    return mae, psnr


def load(path, dn_max):
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)
    if x.shape[0] != 3:
        raise ValueError(f"{path}: expected 3 bands, got {x.shape}")
    return np.clip(x, 0.0, dn_max) / dn_max


def main():
    ap = argparse.ArgumentParser(
        description="Compare V2 and V3R on the real Guwahati pair by difference-based regions."
    )
    ap.add_argument("--cloudy", default="data/raw/cloudy/guwahati_cloudy_test.tif")
    ap.add_argument("--clear", default="data/raw/clear/guwahati_clear.tif")
    ap.add_argument("--v2", default="data/eval/guwahati_liss4_dsen2cr_v2.tif")
    ap.add_argument("--v3r", default="data/eval/guwahati_liss4_dsen2cr_v3r.tif")
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 25, 50, 100])
    args = ap.parse_args()

    c = load(args.cloudy, args.dn_max)
    t = load(args.clear, args.dn_max)
    v2 = load(args.v2, args.dn_max)
    v3r = load(args.v3r, args.dn_max)

    for name, x in [("Cloudy", c), ("V2", v2), ("V3R", v3r)]:
        if x.shape != t.shape:
            raise ValueError(f"{name} shape {x.shape} != target shape {t.shape}")

    diff_dn = np.max(np.abs(c - t), axis=0) * args.dn_max
    all_mask = np.ones(c.shape[1:], dtype=bool)

    base_mae, base_psnr = metric(c, t, all_mask)

    print("=== GUWAHATI LISS-IV V2 / V3R REGION ANALYSIS ===")
    print("Proxy = max per-pixel |cloudy-clear| across G/R/NIR bands")
    print("IMPORTANT: proxy is not a true cloud mask.")
    print(
        f"Cloudy input: MAE={base_mae:.6f} PSNR={base_psnr:.3f} dB"
    )
    print()

    for thr in args.thresholds:
        reg = diff_dn >= thr
        non = ~reg

        base_r = metric(c, t, reg)
        base_n = metric(c, t, non)

        print(f"--- Threshold >= {thr:g} DN ---")
        print(f"Proxy coverage: {reg.mean() * 100:.2f}%")

        for name, x in [("V2", v2), ("V3R", v3r)]:
            mae_r, psnr_r = metric(x, t, reg)
            mae_n, psnr_n = metric(x, t, non)

            print(
                f"{name:<3} region MAE={mae_r:.6f} PSNR={psnr_r:.3f} "
                f"gain={psnr_r-base_r[1]:+.3f} | "
                f"non-region MAE={mae_n:.6f} PSNR={psnr_n:.3f} "
                f"gain={psnr_n-base_n[1]:+.3f}"
            )

        print(
            f"V3R vs V2: region PSNR delta="
            f"{metric(v3r, t, reg)[1]-metric(v2, t, reg)[1]:+.3f} dB | "
            f"non-region PSNR delta="
            f"{metric(v3r, t, non)[1]-metric(v2, t, non)[1]:+.3f} dB"
        )
        print()

    print(
        "Interpretation: the proxy is a cloudy-vs-clear difference mask, not "
        "ground-truth cloud labeling. Guwahati is a temporal pair, so "
        "acquisition differences remain in the metrics."
    )


if __name__ == "__main__":
    main()
