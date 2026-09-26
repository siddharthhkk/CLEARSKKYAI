import argparse
import math

import numpy as np
import rasterio


def psnr(a, b, peak):
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(peak / math.sqrt(mse))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser(
        description="Measure the historical-clear baseline for one LISS-IV pair."
    )
    ap.add_argument("cloudy")
    ap.add_argument("clear")
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    with rasterio.open(args.cloudy) as csrc, rasterio.open(args.clear) as tsrc:
        c = csrc.read().astype(np.float32)
        t = tsrc.read().astype(np.float32)

        if c.shape != t.shape or c.shape[0] != 3:
            raise ValueError(f"Expected identical [3,H,W] arrays, got {c.shape} and {t.shape}")

    c01 = np.clip(c / args.dn_max, 0.0, 1.0)
    t01 = np.clip(t / args.dn_max, 0.0, 1.0)

    print("=== LISS-IV CLOUDY INPUT BASELINE ===")
    print("This is a lower-bound/reference measurement, not model performance.")
    print(f"DN max : {args.dn_max}")
    print(f"MAE    : {mae(c01, t01):.6f}")
    print(f"PSNR   : {psnr(c01, t01, 1.0):.3f} dB")

    for i, name in enumerate(("Green", "Red", "NIR")):
        print(
            f"{name:5s} MAE={mae(c01[i], t01[i]):.6f} "
            f"PSNR={psnr(c01[i], t01[i], 1.0):.3f} dB"
        )

    print(
        "A trained model should be evaluated against this same clear target "
        "using the same normalization. Do not interpret this baseline as a "
        "cloud-only score because the pair also contains temporal/illumination differences."
    )


if __name__ == "__main__":
    main()
