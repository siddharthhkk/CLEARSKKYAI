import argparse
import math
import os

import numpy as np
import rasterio


def metric(pred, target, mask):
    if not np.any(mask):
        return float("nan"), float("nan"), 0

    d = pred[:, mask] - target[:, mask]
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    return mae, psnr, int(mask.sum())


def load(path, dn_max):
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)
    if x.shape[0] != 3:
        raise ValueError(f"{path}: expected 3 bands, got {x.shape}")
    return np.clip(x, 0.0, dn_max) / dn_max


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Analyze real Guwahati LISS-IV V1/V2 outputs inside and outside "
            "difference-based cloud-region proxies."
        )
    )
    ap.add_argument(
        "--cloudy",
        default="data/raw/cloudy/guwahati_cloudy_test.tif",
    )
    ap.add_argument(
        "--v1",
        default="data/eval/guwahati_liss4_dsen2cr.tif",
    )
    ap.add_argument(
        "--v2",
        default="data/eval/guwahati_liss4_dsen2cr_v2.tif",
    )
    ap.add_argument(
        "--clear",
        default="data/raw/clear/guwahati_clear.tif",
    )
    ap.add_argument(
        "--dn-max",
        type=float,
        default=1023.0,
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Difference thresholds in native DN units.",
    )
    args = ap.parse_args()

    c = load(args.cloudy, args.dn_max)
    v1 = load(args.v1, args.dn_max)
    v2 = load(args.v2, args.dn_max)
    t = load(args.clear, args.dn_max)

    shape = t.shape
    for name, x in [("cloudy", c), ("V1", v1), ("V2", v2)]:
        if x.shape != shape:
            raise ValueError(f"{name} shape {x.shape} != target shape {shape}")

    # Conservative proxy: maximum absolute cross-band difference between
    # cloudy and clear acquisitions. This is NOT a true cloud mask.
    diff_dn = np.max(np.abs(c - t), axis=0) * args.dn_max

    print("=== GUWAHATI LISS-IV REGION ANALYSIS ===")
    print("Proxy = max per-pixel |cloudy-clear| across G/R/NIR bands")
    print("IMPORTANT: this is a difference-based proxy, not a true cloud mask.")
    print()

    for thr in args.thresholds:
        reg = diff_dn >= thr
        non = ~reg

        print(f"--- Threshold >= {thr:g} DN ---")
        print(f"Proxy coverage: {reg.mean() * 100:.2f}%")

        for name, x in [
            ("Cloudy", c),
            ("V1", v1),
            ("V2", v2),
        ]:
            mae_r, psnr_r, n_r = metric(x, t, reg)
            mae_n, psnr_n, n_n = metric(x, t, non)
            print(
                f"{name:<6} region MAE={mae_r:.6f} PSNR={psnr_r:.3f} dB "
                f"| non-region MAE={mae_n:.6f} PSNR={psnr_n:.3f} dB "
                f"| pixels={n_r}/{n_n}"
            )

        b_r = metric(c, t, reg)[1]
        v1_r = metric(v1, t, reg)[1]
        v2_r = metric(v2, t, reg)[1]

        b_n = metric(c, t, non)[1]
        v1_n = metric(v1, t, non)[1]
        v2_n = metric(v2, t, non)[1]

        print(
            f"Region PSNR gain: V1={v1_r - b_r:+.3f} dB "
            f"V2={v2_r - b_r:+.3f} dB"
        )
        print(
            f"Non-region PSNR gain: V1={v1_n - b_n:+.3f} dB "
            f"V2={v2_n - b_n:+.3f} dB"
        )
        print()

    print("Done.")
    print(
        "Interpret these numbers as diagnostic evidence only; temporal, "
        "illumination, registration, and other acquisition differences can "
        "also create large cloudy-vs-clear differences."
    )


if __name__ == "__main__":
    main()
