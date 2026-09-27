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
            "Test continuous spatial gating of the existing V2 residual. "
            "Alpha ramps from 0 below t0 to 1 above t1."
        )
    )
    ap.add_argument("--cloudy", default="data/raw/cloudy/guwahati_cloudy_test.tif")
    ap.add_argument("--clear", default="data/raw/clear/guwahati_clear.tif")
    ap.add_argument("--v2", default="data/eval/guwahati_liss4_dsen2cr_v2.tif")
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument(
        "--ramps",
        type=float,
        nargs="+",
        default=[5, 8, 6, 10, 8, 12],
        help="Pairs of t0,t1 DN thresholds. Alpha ramps linearly 0->1.",
    )
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if len(args.ramps) % 2 != 0:
        raise ValueError("--ramps must contain pairs of t0,t1 values.")

    c = load(args.cloudy, args.dn_max)
    t = load(args.clear, args.dn_max)
    v2 = load(args.v2, args.dn_max)

    if c.shape != t.shape or v2.shape != t.shape:
        raise ValueError("Cloudy, V2, and clear images must have identical shapes.")

    residual = v2 - c
    residual_dn = np.max(np.abs(residual), axis=0) * args.dn_max

    # Evaluation-only proxy; never used by the gate.
    diff_dn = np.max(np.abs(c - t), axis=0) * args.dn_max

    all_mask = np.ones(c.shape[1:], dtype=bool)
    base_mae, base_psnr = metric(c, t, all_mask)

    print("=== V2 CONTINUOUS RESIDUAL-GATE SWEEP ===")
    print("Alpha ramps from 0 to 1 based only on V2 residual magnitude.")
    print(f"Cloudy input: MAE={base_mae:.6f} PSNR={base_psnr:.3f} dB")
    print()

    for i in range(0, len(args.ramps), 2):
        t0 = float(args.ramps[i])
        t1 = float(args.ramps[i + 1])

        if t0 < 0 or t1 <= t0:
            raise ValueError(
                f"Invalid ramp ({t0}, {t1}); require 0 <= t0 < t1."
            )

        alpha = np.clip(
            (residual_dn - t0) / (t1 - t0),
            0.0,
            1.0,
        ).astype(np.float32)

        pred = np.clip(c + residual * alpha[None], 0.0, 1.0)

        mae, psnr = metric(pred, t, all_mask)

        strong = alpha >= 0.999
        nonzero = alpha > 0.0

        print(f"--- ramp {t0:g} -> {t1:g} DN ---")
        print(f"Nonzero coverage : {nonzero.mean() * 100:.3f}%")
        print(f"Full-alpha coverage: {strong.mean() * 100:.3f}%")
        print(
            f"Global       MAE={mae:.6f} PSNR={psnr:.3f} dB "
            f"gain={psnr-base_psnr:+.3f} dB | "
            f"MAE change={mae-base_mae:+.6f}"
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

        if not args.no_save:
            tag0 = str(t0).replace(".", "p")
            tag1 = str(t1).replace(".", "p")
            out_path = os.path.join(
                "data",
                "eval",
                f"guwahati_liss4_v2_gate_soft_{tag0}_{tag1}.tif",
            )
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
        "IMPORTANT: gate uses only cloudy input + V2 output. "
        "The clear image is used only for diagnostics."
    )
    print(
        "The Guwahati pair is temporal, so metrics include acquisition "
        "differences beyond cloud effects."
    )


if __name__ == "__main__":
    main()
