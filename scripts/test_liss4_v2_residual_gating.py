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
        description=(
            "Test spatially adaptive V2 residual gating using only the "
            "V2 prediction and cloudy input."
        )
    )
    ap.add_argument(
        "--cloudy",
        default="data/raw/cloudy/guwahati_cloudy_test.tif",
    )
    ap.add_argument(
        "--clear",
        default="data/raw/clear/guwahati_clear.tif",
    )
    ap.add_argument(
        "--v2",
        default="data/eval/guwahati_liss4_dsen2cr_v2.tif",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[5, 10, 20, 30],
        help="Residual magnitude thresholds in native DN units.",
    )
    ap.add_argument("--low-alpha", type=float, default=0.25)
    ap.add_argument("--high-alpha", type=float, default=1.00)
    args = ap.parse_args()

    c = load(args.cloudy, args.dn_max)
    t = load(args.clear, args.dn_max)
    v2 = load(args.v2, args.dn_max)

    if c.shape != t.shape or v2.shape != t.shape:
        raise ValueError("Cloudy, V2, and clear images must have identical shapes.")

    residual = v2 - c

    # Deployment-safe gate: it uses only the cloudy input and V2 prediction.
    residual_dn = np.max(np.abs(residual), axis=0) * args.dn_max

    # Diagnostic proxy for analysis only; it is never used to form the gate.
    diff_dn = np.max(np.abs(c - t), axis=0) * args.dn_max

    all_mask = np.ones(c.shape[1:], dtype=bool)
    base_mae, base_psnr = metric(c, t, all_mask)

    print("=== V2 SPATIAL RESIDUAL GATING EXPERIMENT ===")
    print("Gate uses max per-pixel |V2 - cloudy| across G/R/NIR bands.")
    print(
        f"Low residual  -> alpha={args.low_alpha:.2f}; "
        f"high residual -> alpha={args.high_alpha:.2f}"
    )
    print(f"Cloudy input: MAE={base_mae:.6f} PSNR={base_psnr:.3f} dB")
    print()

    for thr in args.thresholds:
        gate = residual_dn >= thr
        alpha = np.where(
            gate,
            args.high_alpha,
            args.low_alpha,
        ).astype(np.float32)

        pred = np.clip(
            c + residual * alpha[None],
            0.0,
            1.0,
        )

        mae, psnr = metric(pred, t, all_mask)

        print(f"--- residual threshold >= {thr:g} DN ---")
        print(f"Gate coverage: {gate.mean() * 100:.2f}%")
        print(
            f"Global       MAE={mae:.6f} PSNR={psnr:.3f} dB "
            f"gain={psnr-base_psnr:+.3f} dB"
        )

        for proxy_thr in [10, 25, 50, 100]:
            reg = diff_dn >= proxy_thr
            non = ~reg

            mae_r, psnr_r = metric(pred, t, reg)
            mae_n, psnr_n = metric(pred, t, non)

            base_r = metric(c, t, reg)[1]
            base_n = metric(c, t, non)[1]

            print(
                f"  proxy >= {proxy_thr:3d} DN | "
                f"region MAE={mae_r:.6f} PSNR={psnr_r:.3f} "
                f"gain={psnr_r-base_r:+.3f} | "
                f"non-region MAE={mae_n:.6f} PSNR={psnr_n:.3f} "
                f"gain={psnr_n-base_n:+.3f}"
            )

        stem = f"guwahati_liss4_v2_gate_{thr:g}dn"
        out_path = os.path.join("data", "eval", f"{stem}.tif")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(args.v2) as src:
            profile = src.profile.copy()

        profile.update(dtype="uint16", count=3, compress="deflate")
        out_dn = np.clip(
            np.rint(pred * args.dn_max),
            0,
            args.dn_max,
        ).astype(np.uint16)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_dn)

        print(f"  saved: {os.path.abspath(out_path)}")
        print()

    print(
        "IMPORTANT: the gate is deployment-safe because it uses only the "
        "cloudy input and V2 output. The evaluation region proxy uses the "
        "clear image only for diagnostics."
    )
    print(
        "The Guwahati pair is temporal, so global and regional metrics can "
        "include illumination, registration, and other acquisition differences."
    )


if __name__ == "__main__":
    main()
