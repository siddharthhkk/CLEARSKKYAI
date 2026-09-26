import argparse
import os

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser(
        description="Quantify cloudy vs historical-clear LISS-IV differences."
    )
    ap.add_argument("cloudy")
    ap.add_argument("clear")
    args = ap.parse_args()

    with rasterio.open(args.cloudy) as csrc, rasterio.open(args.clear) as tsrc:
        if (
            csrc.count != 3
            or tsrc.count != 3
            or csrc.shape != tsrc.shape
            or csrc.transform != tsrc.transform
            or csrc.crs != tsrc.crs
        ):
            raise ValueError("Cloudy and clear rasters must share the same 3-band grid.")

        c = csrc.read().astype(np.float32)
        t = tsrc.read().astype(np.float32)

    diff = np.abs(c - t)

    print("=== LISS-IV PAIR DIFFERENCE DIAGNOSTIC ===")
    print("This is a diagnostic, not a cloud mask or benchmark metric.")
    for i, name in enumerate(("Green", "Red", "NIR")):
        corr = float(np.corrcoef(c[i].ravel(), t[i].ravel())[0, 1])
        print(
            f"{name:5s}: cloudy mean={c[i].mean():.3f} "
            f"clear mean={t[i].mean():.3f} "
            f"MAE={diff[i].mean():.3f} "
            f"p95_abs_diff={np.percentile(diff[i],95):.3f} "
            f"corr={corr:.4f}"
        )

    pooled = diff.mean(axis=0)
    for threshold in (10, 25, 50, 100):
        pct = float((pooled >= threshold).mean() * 100.0)
        print(f"Pixels with mean |cloudy-clear| >= {threshold:3d}: {pct:6.2f}%")

    print(
        "\nInterpretation: large differences can contain clouds, shadows, "
        "seasonal vegetation change, illumination differences, or registration "
        "artifacts. Do not treat them as a true cloud mask."
    )


if __name__ == "__main__":
    main()
