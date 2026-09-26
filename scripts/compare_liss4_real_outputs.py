import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def metrics(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    return mae, psnr


def stretch(images):
    vals = []
    for x in images:
        for i in range(3):
            vals.append(x[i].ravel())
    vals = np.concatenate(vals)

    out = []
    for x in images:
        y = np.empty_like(x, dtype=np.float32)
        for i in range(3):
            lo, hi = np.percentile(
                np.concatenate([x[i].ravel(), vals]),
                (2, 98),
            )
            if hi <= lo:
                y[i] = 0.0
            else:
                y[i] = np.clip((x[i] - lo) / (hi - lo), 0.0, 1.0)
        out.append(y)
    return out


def fcc(x):
    # LISS-IV bands are [Green, Red, NIR].
    # Display as false-colour [NIR, Red, Green].
    return np.transpose(x[[2, 1, 0]], (1, 2, 0))


def main():
    ap = argparse.ArgumentParser(
        description="Compare cloudy input, V1, V2, and clear Guwahati LISS-IV outputs."
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
        "--output",
        default="data/eval/guwahati_liss4_v1_v2_compare.png",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    paths = {
        "Cloudy": args.cloudy,
        "V1": args.v1,
        "V2": args.v2,
        "Clear": args.clear,
    }

    imgs = {}
    for name, path in paths.items():
        with rasterio.open(path) as src:
            x = src.read().astype(np.float32)
            if x.shape[0] != 3:
                raise ValueError(f"{name}: expected 3 bands, got {x.shape}")
            imgs[name] = np.clip(x, 0.0, args.dn_max) / args.dn_max

    ref = imgs["Clear"]
    shape = ref.shape
    for name, x in imgs.items():
        if x.shape != shape:
            raise ValueError(
                f"{name}: shape {x.shape} does not match Clear {shape}"
            )

    for name in ("Cloudy", "V1", "V2"):
        mae, psnr = metrics(imgs[name], ref)
        print(f"{name:<6} MAE={mae:.6f}  PSNR={psnr:.3f} dB")

    print()
    base_psnr = metrics(imgs["Cloudy"], ref)[1]
    for name in ("V1", "V2"):
        psnr = metrics(imgs[name], ref)[1]
        print(f"{name:<6} PSNR gain vs cloudy = {psnr - base_psnr:+.3f} dB")

    display, = stretch([imgs["Clear"]])
    # Use one common scale for each respective image so comparisons are not
    # driven by independent contrast stretching.
    def disp(x):
        y = np.empty_like(x)
        for i in range(3):
            lo, hi = np.percentile(
                np.concatenate([v[i].ravel() for v in imgs.values()]),
                (2, 98),
            )
            y[i] = np.clip((x[i] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return y

    panel = {k: disp(v) for k, v in imgs.items()}

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    order = ["Cloudy", "V1", "V2", "Clear"]

    for j, name in enumerate(order):
        mae, psnr = metrics(imgs[name], ref)
        title = name
        if name != "Clear":
            title += f"\nPSNR {psnr:.2f} dB | MAE {mae:.4f}"
        ax[0, j].imshow(fcc(panel[name]))
        ax[0, j].set_title(title)

    for j, name in enumerate(order):
        if name == "Clear":
            ax[1, j].imshow(fcc(panel[name]))
            ax[1, j].set_title("Clear reference")
        else:
            err = np.mean(np.abs(imgs[name] - ref), axis=0)
            ax[1, j].imshow(err, cmap="magma")
            ax[1, j].set_title(f"{name} absolute error")

    for a in ax.ravel():
        a.axis("off")

    fig.suptitle("Guwahati LISS-IV: Cloudy vs V1 vs V2 vs Clear", fontsize=14)
    fig.tight_layout()

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison: {out}")


if __name__ == "__main__":
    main()
