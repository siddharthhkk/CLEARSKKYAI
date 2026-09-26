import argparse
import os
import sys

import numpy as np
import rasterio
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR


def metrics(pred, target):
    d = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else float("inf")
    return mae, psnr


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a synthetic-pretrained LISS-IV checkpoint on the held-out real Guwahati pair."
    )
    ap.add_argument("--checkpoint", default="weights/liss4_dsen2cr_synthetic.pth")
    ap.add_argument("--cloudy", default="data/raw/cloudy/guwahati_cloudy_test.tif")
    ap.add_argument("--clear", default="data/raw/clear/guwahati_clear.tif")
    ap.add_argument("--output", default="data/eval/guwahati_liss4_dsen2cr.tif")
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument("--tile", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = LISS4DSen2CR(
        features=int(ckpt.get("features", 256)),
        blocks=int(ckpt.get("blocks", 16)),
        res_scale=0.1,
        use_sar=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with rasterio.open(args.cloudy) as csrc, rasterio.open(args.clear) as tsrc:
        if csrc.count != 3 or tsrc.count != 3:
            raise ValueError("Cloudy and clear inputs must both have 3 bands.")
        if csrc.width != tsrc.width or csrc.height != tsrc.height:
            raise ValueError("Cloudy and clear dimensions differ.")
        if csrc.crs != tsrc.crs or csrc.transform != tsrc.transform:
            raise ValueError("Cloudy and clear grids are not identical.")

        h, w = csrc.height, csrc.width
        cloudy_full = csrc.read().astype(np.float32)
        clear_full = tsrc.read().astype(np.float32)

        out = np.zeros_like(clear_full, dtype=np.float32)

        for row in range(0, h, args.tile):
            for col in range(0, w, args.tile):
                hh = min(args.tile, h - row)
                ww = min(args.tile, w - col)
                x = cloudy_full[:, row:row + hh, col:col + ww]

                if hh != args.tile or ww != args.tile:
                    py = args.tile - hh
                    px = args.tile - ww
                    x = np.pad(x, ((0, 0), (0, py), (0, px)), mode="edge")

                xt = normalize_liss4(x, args.dn_max)[None].to(
                    device, non_blocking=True
                )

                with torch.no_grad():
                    y = model(xt)
                    y = torch.clamp(y, 0.0, 1.0)

                y = (y[0].detach().cpu().numpy()[:, :hh, :ww] * args.dn_max)
                out[:, row:row + hh, col:col + ww] = y

        base = np.clip(cloudy_full, 0.0, args.dn_max) / args.dn_max
        target = np.clip(clear_full, 0.0, args.dn_max) / args.dn_max
        pred = np.clip(out, 0.0, args.dn_max) / args.dn_max

        base_mae, base_psnr = metrics(base, target)
        pred_mae, pred_psnr = metrics(pred, target)

        print("=== REAL GUWAHATI LISS-IV EVALUATION ===")
        print(f"Device        : {device}")
        if device.type == "cuda":
            print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"Checkpoint    : {os.path.abspath(args.checkpoint)}")
        print(f"Best synth PSNR: {float(ckpt.get('best_val_psnr', float('nan'))):.3f} dB")
        print(f"Input  MAE    : {base_mae:.6f}")
        print(f"Output MAE    : {pred_mae:.6f}")
        print(f"Input  PSNR   : {base_psnr:.3f} dB")
        print(f"Output PSNR   : {pred_psnr:.3f} dB")
        print(f"PSNR gain     : {pred_psnr - base_psnr:+.3f} dB")
        print(f"MAE change    : {pred_mae - base_mae:+.6f}")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        profile = csrc.profile.copy()
        profile.update(dtype="uint16", count=3, compress="deflate")
        out_dn = np.clip(np.rint(out), 0, args.dn_max).astype(np.uint16)
        with rasterio.open(args.output, "w", **profile) as dst:
            dst.write(out_dn)

        print(f"Output image  : {os.path.abspath(args.output)}")
        print(
            "IMPORTANT: Guwahati is a real cloudy/cloud-free temporal pair. "
            "These global metrics mix cloud removal with temporal, illumination, "
            "registration, and other scene differences."
        )
        print(
            "This is a held-out real-cloud evaluation of the synthetic-pretrained model, "
            "not a multi-scene benchmark."
        )


if __name__ == "__main__":
    main()
