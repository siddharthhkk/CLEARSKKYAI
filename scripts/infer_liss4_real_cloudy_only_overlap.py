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


def make_weight(h, w):
    """
    Smooth positive center-weighted window for overlap blending.

    Clip the sine argument before sqrt because floating-point evaluation of
    sin(pi) can produce a tiny negative number at the last pixel.
    """
    y = np.clip(np.sin(np.pi * np.linspace(0.0, 1.0, h, dtype=np.float32)), 0.0, 1.0)
    x = np.clip(np.sin(np.pi * np.linspace(0.0, 1.0, w, dtype=np.float32)), 0.0, 1.0)

    wy = 0.15 + 0.85 * np.sqrt(y)
    wx = 0.15 + 0.85 * np.sqrt(x)

    return (wy[:, None] * wx[None, :]).astype(np.float32)


def tile_starts(length, tile, stride):
    """Return starts that cover the entire axis, including the far edge."""
    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)

    return starts


def main():
    ap = argparse.ArgumentParser(
        description="Cloudy-only LISS-IV inference with overlapping tile blending."
    )
    ap.add_argument(
        "--checkpoint",
        default="weights/liss4_dsen2cr_synthetic_v3r.pth",
    )
    ap.add_argument("--cloudy", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between neighboring tiles in pixels.",
    )
    args = ap.parse_args()

    if args.tile <= 0:
        raise ValueError("--tile must be positive.")
    if not 0 <= args.overlap < args.tile:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile.")

    stride = args.tile - args.overlap

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
        x_full = src.read().astype(np.float32)
        profile = src.profile.copy()

    rs = tile_starts(h, args.tile, stride)
    cs = tile_starts(w, args.tile, stride)

    accum = np.zeros_like(x_full, dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    tile_count = 0

    for row in rs:
        for col in cs:
            row2 = min(row + args.tile, h)
            col2 = min(col + args.tile, w)

            hh = row2 - row
            ww = col2 - col
            x = x_full[:, row:row2, col:col2]

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
                y = torch.clamp(model(xt), 0.0, 1.0)

            pred = y[0].detach().cpu().numpy()[:, :hh, :ww]
            wt = make_weight(hh, ww)

            accum[:, row:row2, col:col2] += pred * wt[None]
            weights[row:row2, col:col2] += wt

            tile_count += 1

    if not np.all(weights > 0):
        raise RuntimeError(
            "Overlap blending left uncovered pixels. Check tile/overlap settings."
        )

    out = accum / weights[None]

    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite values detected in blended output.")

    out_dn = np.clip(
        np.rint(out * args.dn_max),
        0,
        args.dn_max,
    ).astype(np.uint16)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    profile.update(
        dtype="uint16",
        count=3,
        compress="deflate",
    )

    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(out_dn)

    print("=== REAL LISS-IV OVERLAP-TILED INFERENCE ===")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint : {os.path.abspath(args.checkpoint)}")
    print(f"Input      : {os.path.abspath(args.cloudy)}")
    print(f"Output     : {os.path.abspath(args.output)}")
    print(f"Size       : {w} x {h}")
    print(f"Tile       : {args.tile} px")
    print(f"Overlap    : {args.overlap} px")
    print(f"Tiles      : {tile_count} ({len(rs)} rows x {len(cs)} cols)")
    print()
    print(
        "No clear reference was used. Overlap blending reduces tile-boundary "
        "artifacts for qualitative cross-scene evaluation."
    )


if __name__ == "__main__":
    main()
