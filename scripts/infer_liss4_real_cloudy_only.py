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


def main():
    ap = argparse.ArgumentParser(
        description="Run LISS-IV inference on a real cloudy GeoTIFF without requiring a clear reference."
    )
    ap.add_argument(
        "--checkpoint",
        default="weights/liss4_dsen2cr_synthetic_v3r.pth",
    )
    ap.add_argument(
        "--cloudy",
        required=True,
    )
    ap.add_argument(
        "--output",
        required=True,
    )
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

    with rasterio.open(args.cloudy) as src:
        if src.count != 3:
            raise ValueError(
                f"Expected 3-band LISS-IV GeoTIFF, got {src.count} bands."
            )

        h, w = src.height, src.width
        profile = src.profile.copy()
        x_full = src.read().astype(np.float32)

        out = np.zeros_like(x_full, dtype=np.float32)

        for row in range(0, h, args.tile):
            for col in range(0, w, args.tile):
                hh = min(args.tile, h - row)
                ww = min(args.tile, w - col)

                x = x_full[:, row:row + hh, col:col + ww]

                if hh != args.tile or ww != args.tile:
                    py = args.tile - hh
                    px = args.tile - ww
                    x = np.pad(
                        x,
                        ((0, 0), (0, py), (0, px)),
                        mode="edge",
                    )

                xt = normalize_liss4(x, args.dn_max)[None].to(
                    device,
                    non_blocking=True,
                )

                with torch.no_grad():
                    y = model(xt)
                    y = torch.clamp(y, 0.0, 1.0)

                out[
                    :,
                    row:row + hh,
                    col:col + ww,
                ] = (
                    y[0]
                    .detach()
                    .cpu()
                    .numpy()[:, :hh, :ww]
                    * args.dn_max
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    profile.update(
        dtype="uint16",
        count=3,
        compress="deflate",
    )

    out_dn = np.clip(
        np.rint(out),
        0,
        args.dn_max,
    ).astype(np.uint16)

    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(out_dn)

    print("=== REAL LISS-IV CLOUDY-ONLY INFERENCE ===")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint : {os.path.abspath(args.checkpoint)}")
    print(f"Input      : {os.path.abspath(args.cloudy)}")
    print(f"Output     : {os.path.abspath(args.output)}")
    print(f"Size       : {w} x {h}")
    print()
    print(
        "No ground-truth clear image was used. This is qualitative "
        "cross-scene inference only; no PSNR/MAE claim is made."
    )


if __name__ == "__main__":
    main()
