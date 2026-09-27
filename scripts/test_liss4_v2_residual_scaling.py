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
            "Evaluate residual scaling on the existing V2 LISS-IV output "
            "without retraining."
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
    ap.add_argument(
        "--dn-max",
        type=float,
        default=1023.0,
    )
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
    )
    args = ap.parse_args()

    c = load(args.cloudy, args.dn_max)
    t = load(args.clear, args.dn_max)
    v2 = load(args.v2, args.dn_max)

    if c.shape != t.shape or v2.shape != t.shape:
        raise ValueError("Cloudy, V2, and clear images must have identical shapes.")

    # Recover the learned residual from the saved V2 output:
    # residual = V2_output - cloudy.
    residual = v2 - c

    print("=== V2 RESIDUAL SCALING EXPERIMENT ===")
    print("No retraining; scales the existing V2 correction:")
    print("    output_alpha = cloudy + alpha * (V2 - cloudy)")
    print("alpha=1.0 reproduces the V2 output.")
    print()

    diff_dn = np.max(np.abs(c - t), axis=0) * args.dn_max

    # Global input metrics for reference.
    mae0, psnr0 = metric(c, t, np.ones(c.shape[1:], dtype=bool))
    print(f"Cloudy input: MAE={mae0:.6f} PSNR={psnr0:.3f} dB")
    print()

    for a in args.alphas:
        if a < 0:
            raise ValueError("Alpha must be >= 0.")

        pred = np.clip(c + a * residual, 0.0, 1.0)

        mae, psnr = metric(
            pred,
            t,
            np.ones(c.shape[1:], dtype=bool),
        )

        print(
            f"--- alpha={a:.2f} ---\n"
            f"Global       MAE={mae:.6f} PSNR={psnr:.3f} dB "
            f"gain={psnr-psnr0:+.3f} dB"
        )

        for thr in [10, 25, 50, 100]:
            reg = diff_dn >= thr
            non = ~reg

            mae_r, psnr_r = metric(pred, t, reg)
            mae_n, psnr_n = metric(pred, t, non)

            base_r = metric(c, t, reg)[1]
            base_n = metric(c, t, non)[1]

            print(
                f"  >= {thr:3d} DN | "
                f"region MAE={mae_r:.6f} PSNR={psnr_r:.3f} "
                f"gain={psnr_r-base_r:+.3f} | "
                f"non-region MAE={mae_n:.6f} PSNR={psnr_n:.3f} "
                f"gain={psnr_n-base_n:+.3f}"
            )

        # Save each scaled image for visual inspection.
        stem = f"guwahati_liss4_v2_alpha_{a:.2f}".replace(".", "p")
        out_path = os.path.join(
            "data",
            "eval",
            f"{stem}.tif",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(
            args.v2,
        ) as src:
            profile = src.profile.copy()

        profile.update(dtype="uint16", count=3, compress="deflate")
        out_dn = np.clip(np.rint(pred * args.dn_max), 0, args.dn_max).astype(
            np.uint16
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_dn)

        print(f"  saved: {os.path.abspath(out_path)}")
        print()

    print(
        "IMPORTANT: the region proxy is based on cloudy-vs-clear differences, "
        "not a true cloud mask. Guwahati is also a temporal pair, so metrics "
        "include non-cloud acquisition differences."
    )


if __name__ == "__main__":
    main()
